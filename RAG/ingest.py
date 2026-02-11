import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ⭐ Correct folder path
PDF_DIR = "../DATA"

documents = []

# Load PDFs
for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_DIR, file)

        print("Loading:", file)

        loader = PyPDFLoader(path)
        docs = loader.load()

        # Add metadata
        for d in docs:
            d.metadata["source"] = file

        documents.extend(docs)

print("Loaded pages:", len(documents))


# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# Embeddings
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
db = FAISS.from_documents(chunks, embedder)

# ⭐ Save in correct location
db.save_local("../VectorDB")

print("Vector DB saved!")

print("HI")
