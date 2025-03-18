import os 
import streamlit as st # used to create our UI frontend 
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 

import ollama 
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import Chroma 

st.title('Chat with Document') # title in our web page 
uploaded_file = st.file_uploader('Upload file:', type=['txt','pdf','docx']) 
add_file = st.button('Add File') 

if uploaded_file and add_file: 
    with st.spinner('Reading, chunking and embedding file...'):
        bytes_data = uploaded_file.read() 
        file_name = os.path.join('./', uploaded_file.name) 
        with open(file_name,'wb') as f: 
            f.write(bytes_data) 
        
        name,extension = os.path.splitext(file_name)
        
        if extension == '.pdf': 
            from langchain_community.document_loaders import PyPDFLoader 
            loader = PyPDFLoader(file_name) 
        elif extension == '.docx': 
            from langchain_community.document_loaders import Docx2txtLoader 
            loader = Docx2txtLoader(file_name) 
        elif extension == '.txt': 
            from langchain_community.document_loaders import TextLoader 
            loader = TextLoader(file_name) 
        else: 
            st.write('Document format is not supported!')
        
        # loader = TextLoader(file_name) # to load text document 
        documents = loader.load() 
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
        
        chunks = text_splitter.split_documents(documents) 
        
        ollama.pull('nomic-embed-text') 
        embeddings = OllamaEmbeddings(model='nomic-embed-text') 
        vector_store = Chroma.from_documents(chunks,embeddings) 
        
        from langchain.prompts import ChatPromptTemplate, PromptTemplate 
        from langchain_core.output_parsers import StrOutputParser 
        from langchain_ollama import ChatOllama 
        from langchain_core.runnables import RunnablePassthrough 
        from langchain.retrievers.multi_query import MultiQueryRetriever

        llm = ChatOllama(model="llama3.2") 
        # a simple technique to generate multiple questions from a single question and then retrieve documents 
        # based on those questions, getting the best of both worlds. 
        
        QUERY_PROMPT = PromptTemplate( 
            input_variables=["question"], 
            template="""You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: 
            {question}""", 
        ) 
        
        retriever = MultiQueryRetriever.from_llm( 
            vector_store.as_retriever(), llm, prompt=QUERY_PROMPT 
        ) 
        
        # RAG prompt 
        template = """Answer the question based ONLY on the following context: 
        {context} 
        Question: {question} 
        """ 
        
        prompt = ChatPromptTemplate.from_template(template) 
        
        chain = ( 
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser() 
        ) 
        
        st.session_state.chain = chain
        st.success('File uploaded, chunked and embedded successfully') 
    
question = st.text_input('Input your question') 

if question: 
    if 'chain' in st.session_state: 
        chain = st.session_state.chain 
        res = chain.invoke(input=(question)) 
        st.write(res)


