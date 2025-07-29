import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import shutil

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file. Please set it.")
    exit()

# Configuration
DATA_PATH = "data/policies"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

def main():
    print("Starting data ingestion process...")

    # Create embedding function
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Load documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Clear out the database first if it already exists
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing ChromaDB at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
        print("Existing ChromaDB removed.")

    # Create a new ChromaDB instance and persist it
    print("Creating/updating ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector store persisted successfully.")
    print("Data ingestion complete.")

if __name__ == "__main__":
    main()