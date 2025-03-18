import os
from openai import OpenAI
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize DeepSeek LLM and Embeddings
deepseek_llm = OpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)
deepseek_embeddings = OpenAIEmbeddings(
    model="deepseek-embedding",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com"
)

# Streamlit UI
st.title('Chat with Multiple Documents (DeepSeek RAG)')

# File Upload
uploaded_files = st.file_uploader('Upload files:', type=['txt', 'pdf', 'docx'], accept_multiple_files=True)
add_files = st.button('Add Files')

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state['documents'] = []

if uploaded_files and add_files:
    with st.spinner('Processing files...'):
        new_documents = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join('./', uploaded_file.name)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            _, extension = os.path.splitext(file_path)
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif extension == '.txt':
                loader = TextLoader(file_path)
            else:
                st.write(f'Unsupported format: {extension}')
                continue
            
            documents = loader.load()
            new_documents.extend(documents)
        
        st.session_state['documents'].extend(new_documents)

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(st.session_state['documents'])

        # Vector store
        vector_store = Chroma.from_documents(chunks, deepseek_embeddings, persist_directory="./chroma_db")
        vector_store.persist()

        # Multi-query retriever
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Generate five different versions of the given user question 
            to retrieve relevant documents. Provide alternatives separated by newlines.
            Original question: {question}"""
        )
        
        retriever = MultiQueryRetriever.from_llm(vector_store.as_retriever(), deepseek_llm, prompt=QUERY_PROMPT)

        # RAG Chain
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | deepseek_llm
            | StrOutputParser()
        )

        st.session_state['chain'] = chain
        st.success('Files processed successfully!')

# Question input
question = st.text_input('Ask a question based on the uploaded documents')

if question:
    if 'chain' in st.session_state:
        chain = st.session_state['chain']
        response = chain.invoke(input=question)
        st.write(response)
    else:
        st.write("Please upload files first.")
