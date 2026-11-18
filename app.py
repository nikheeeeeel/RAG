import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Loading PDF..")
loader = PyPDFLoader("NikhilKishore_Resume.pdf")
document = loader.load()

print("Chunking Text..")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)
chunks = text_splitter.split_documents(document)

print("Creating vector DB..")
embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")
vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = "./chroma_db"
)

retriever = vector_db.as_retriever(search_kwargs={"k":5})
template = """
Use the following pieces of retrieved context to answer the question.If you don't know the answer just say you don't know. Use three sentences maximum and keep the answewr concise.
Context: {context}
Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(model = "gemini-3.8-flash", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
)

user_question = "What is APISEC?"

print("\nQuestion:", user_question, flush=True)

print("\n1. Testing retrieval...", flush=True)

retrieved_docs = retriever.invoke(user_question)

print("2. Retrieval completed!", flush=True)
print("Retrieved chunks:", len(retrieved_docs))

for doc in retrieved_docs:
    print(doc.page_content)

print("\n3. Sending request to Gemini...", flush=True)

response = rag_chain.invoke(user_question)

print("\n4. Gemini responded!", flush=True)
print("Answer:", response.content, flush=True)
