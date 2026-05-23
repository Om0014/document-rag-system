from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

documents = []

# Load PDFs
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"docs/{file}")
        documents.extend(loader.load())

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store vectors
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("vectorstore")

print("Documents processed successfully!")