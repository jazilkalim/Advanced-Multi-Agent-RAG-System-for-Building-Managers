import pickle
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Optional: clean old store
shutil.rmtree("chroma_store", ignore_errors=True)

# Create embedder
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create and persist DB
db = Chroma.from_documents(
    chunks,
    embedding=embedder,
    persist_directory="chroma_store"
)

print("✅ Vector store created and saved.")
