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
    # This error message is now more accurate for deployment
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in Streamlit Cloud secrets.")

# --- Debugging additions start here ---
print("DEBUG: rag_pipeline.py - Script started.")
print(f"DEBUG: GOOGLE_API_KEY is present (length: {len(GOOGLE_API_KEY) if GOOGLE_API_KEY else 0}).")
# DO NOT print the full API key for security reasons.
# Just checking its presence or length is enough for debugging.

# Configuration
CHROMA_PATH = "chroma_db"
print(f"DEBUG: CHROMA_PATH is set to: {CHROMA_PATH}")

# Define the embedding model
# Explicitly pass the API key here
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
print("DEBUG: Embedding model initialized.")

def get_rag_chain():
    print("DEBUG: Entering get_rag_chain function.")
    try:
        # The temporary chromadb import test is no longer needed as sqlite3 issue is fixed.
        # You can remove or comment out these lines if you added them:
        # import chromadb
        # print("DEBUG: chromadb imported successfully (temporary test).")
        # temp_client = chromadb.Client()
        # print("DEBUG: In-memory Chroma client created successfully (temporary test).")

        # --- Original logic to load from persisted directory ---
        print(f"DEBUG: Attempting to load ChromaDB from: {CHROMA_PATH}")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        print("DEBUG: ChromaDB loaded successfully.")

        # Define the language model
        # Explicitly pass the API key here too
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
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