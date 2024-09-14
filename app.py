import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


load_dotenv()

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", temperature=0.1)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

hybrid_template = """ 
"Please answer the following question based on the provided `context` that follows the question.\n"
"If you do not know the answer then just say 'I do not know'\n"
"question: {question}\n"
"context: ```{context}```\n
"""

PROMPT = PromptTemplate(
    template=hybrid_template, input_variables=["context", "question"]
)

#document preprocessing

def process_document(file):
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    if file.name.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path) #for pdf files
    else:
        loader = UnstructuredImageLoader(file_path, mode="elements") #this is for jpg and jpeg files

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    texts = text_splitter.split_documents(documents)
    return texts

#FAISS Vector store embeddings

def create_vector_stores(texts):
    return FAISS.from_documents(texts, embedding)

#Query Processing and Adaptive RAG

def process_query(vectorstore, query, feedback = None):
    #adaptive RAG
    base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    compressed_vectorstore = FAISS.from_documents(compressed_docs, embedding)

     
    final_retriever = compressed_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    if feedback:
        hybrid_feedback_template = hybrid_template + f"\nPrevious feedback: {feedback}\n. Please consider this feedback while answering."
        PROMPT_WITH_FEEDBACK = PromptTemplate(template=hybrid_feedback_template, input_variables=["context", "question"])
    else:
        PROMPT_WITH_FEEDBACK = PROMPT


    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = final_retriever,
        return_source_documents = True,
        chain_type_kwargs = {"prompt": PROMPT_WITH_FEEDBACK})
    result = qa_chain({"query": query})
    return result['result'], result['source_documents']

#post processing - strip, summarize along with formatted sources
def post_process(answer, sources):
    answer = answer.strip()

    #summarize
    if len(answer) > 500:
        summary_prompt = f"Summarize the following answer in 2-3 sentences: {answer}"
        summary = llm.predict(summary_prompt)
        answer = f"{summary}\n\nFull Answer: {answer}"
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        formatted_source = f"{i}. {source.page_content[:200]}..."
        formatted_sources.append(formatted_source)
    return answer, formatted_sources

#THE ACTUAL FEEDBACK LOOP

def process_feedback(feedback, query, answer):
    feedback_prompt = f"""
    Given the following:
    Question: {query}
    answer: {answer}
    User Feedback: {feedback}

    Please Provide suggestions on how to improve the answer based on the user's feedback:
    """
    improvement_suggestions = llm.predict(feedback_prompt)
    return improvement_suggestions


# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

st.title("Document Q&A Chatbot")

uploaded_file = st.file_uploader("Choose a PDF or Image File", type=["pdf", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner('Processing file... This may take a while for images.'):
        texts = process_document(uploaded_file)
        st.session_state.vectorstore = create_vector_stores(texts)
    st.success('File uploaded and processed successfully!')

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about the document:"):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                answer, sources = process_query(st.session_state.vectorstore, query)
                processed_answer, formatted_sources = post_process(answer, sources)
                
                st.markdown(f"{processed_answer}")
                
                with st.expander("Sources"):
                    for source in formatted_sources:
                        st.markdown(f"- {source}")
        
        st.session_state.chat_history.append({"role": "assistant", "content": processed_answer})
    else:
        st.error("Please upload a document first.")


if st.button('Clear Chat History'):
    st.session_state.chat_history = []
    st.experimental_rerun()

