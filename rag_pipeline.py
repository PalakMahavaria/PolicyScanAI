# rag_pipeline.py

import os
# from dotenv import load_dotenv # Ensure this is still commented out or removed

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.") # Updated message

# Configuration
CHROMA_PATH = "chroma_db"

# --- Debugging additions start here ---
print("DEBUG: rag_pipeline.py - Script started.")
print(f"DEBUG: CHROMA_PATH is set to: {CHROMA_PATH}")

# Define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("DEBUG: Embedding model initialized.")

def get_rag_chain():
    print("DEBUG: Entering get_rag_chain function.")
    try:
        # --- TEMPORARY TEST: Try to import chromadb and create an in-memory instance ---
        # This helps us see if the core chromadb library can even be imported
        # before trying to load your specific persisted data.
        import chromadb # This is the line that was failing before
        print("DEBUG: chromadb imported successfully (temporary test).")
        # Try creating a temporary in-memory client
        temp_client = chromadb.Client()
        print("DEBUG: In-memory Chroma client created successfully (temporary test).")
        # If these lines execute, the issue is with loading from persist_directory.
        # If it fails here, the core chromadb library itself has an issue.

        # --- Original logic to load from persisted directory ---
        print(f"DEBUG: Attempting to load ChromaDB from: {CHROMA_PATH}")
        # This is line 40, where the error was previously pointing
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        print("DEBUG: ChromaDB loaded successfully.")

        # Define the language model
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        print("DEBUG: LLM initialized.")

        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant for insurance policy queries.
            Answer the user's question based ONLY on the provided context.
            If the answer is not found in the context, politely state that you don't have enough information.
            Provide clear and concise answers.
            Context: {context}"""),
            ("human", "{input}")
        ])
        print("DEBUG: Prompt template created.")

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        print("DEBUG: Document chain created.")

        # Create the retriever
        retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant documents
        print("DEBUG: Retriever created.")

        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print("DEBUG: Retrieval chain created.")

        return retrieval_chain

    except Exception as e:
        print(f"ERROR: An error occurred in get_rag_chain: {e}")
        # Re-raise the exception so Streamlit still catches it and shows the traceback
        raise

print("DEBUG: rag_pipeline.py - Script finished setup.")