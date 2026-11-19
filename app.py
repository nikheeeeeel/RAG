import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough



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
embeddings = OllamaEmbeddings(model = "embeddinggemma")
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

llm = ChatOllama(model = "gemma4:e2b", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
)

question = ""
while(question != "exit"):
    question = input("Enter question: ")
    if question == "exit":
        break
    response = rag_chain.invoke(question)
    print("model thinking..")
    print(response.usage_metadata)
    print(response.content)
