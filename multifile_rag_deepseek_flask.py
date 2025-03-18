import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Flask 앱 생성
app = Flask(__name__)

# 파일 저장 폴더
UPLOAD_FOLDER = "./uploads"
VECTORSTORE_FOLDER = "./chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# DeepSeek LLM 및 Embedding 초기화
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)
deepseek_embeddings = OpenAIEmbeddings(
    model="deepseek-embedding",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com"
)

# 문서 데이터 저장
documents = []

@app.route('/upload', methods=['POST'])
def upload_files():
    global documents
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    uploaded_files = request.files.getlist('files')
    new_documents = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(file_path)

        _, extension = os.path.splitext(file_path)
        if extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif extension == '.txt':
            loader = TextLoader(file_path)
        else:
            return jsonify({"error": f"Unsupported format: {extension}"}), 400

        documents.extend(loader.load())

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 벡터스토어 생성 및 저장
    vector_store = Chroma.from_documents(chunks, deepseek_embeddings, persist_directory=VECTORSTORE_FOLDER)
    vector_store.persist()

    return jsonify({"message": "Files uploaded and processed successfully"}), 200


@app.route('/query', methods=['POST'])
def query_rag():
    global documents
    if not documents:
        return jsonify({"error": "No documents uploaded"}), 400

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 벡터스토어 로드
    vector_store = Chroma(persist_directory=VECTORSTORE_FOLDER, embedding_function=deepseek_embeddings)

    # Multi-query retriever 설정
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate five different versions of the given user question 
        to retrieve relevant documents. Provide alternatives separated by newlines.
        Original question: {question}"""
    )
    retriever = MultiQueryRetriever.from_llm(vector_store.as_retriever(), deepseek_llm, prompt=QUERY_PROMPT)

    # RAG 체인
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

    response = chain.invoke(input=question)
    return jsonify({"response": response})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

