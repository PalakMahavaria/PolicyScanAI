import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Configuration
CHROMA_PATH = "chroma_db"

# Prompt template for the RAG chain
PROMPT_TEMPLATE = """
You are an expert assistant for insurance policy analysis. Your task is to answer questions about insurance policies.
Use the provided context strictly to answer the question. If the answer is not in the provided context, state that you cannot answer from the given information.
Provide the answer in a clear, concise, and structured format.
Always cite the source document names (e.g., "Source: [document_name.pdf]") for each piece of information you provide.
If multiple sources contribute to an answer, list all relevant sources.

Context:
{context}

Question: {question}

Answer:
"""

def get_rag_chain():
    # Create embedding function
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Load the Chroma DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant documents

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    # Create the RAG chain
    # 1. Format docs for context
    def format_docs(docs):
        formatted_strings = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown Source")
            formatted_strings.append(f"--- Document {i+1} (Source: {os.path.basename(source)}) ---\n{doc.page_content}")
        return "\n\n".join(formatted_strings)

    # 2. Define the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    print("Testing the RAG pipeline...")
    rag_chain = get_rag_chain()

    # Test queries
    queries = [
        "What is the definition of 'Accident' in the policies?",
        "What are the exclusions for pre-existing diseases?",
        "Can I get coverage for dental treatment?",
        "What is the grace period for policy renewal?",
        "What is not covered under health insurance?",
        "Which policy covers domestic travel?",
        "List items that are subsumed into costs of treatment?",
        "What is the definition of AYUSH Hospital?"
    ]

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: {query} ---")
        response = rag_chain.invoke(query)
        print(response)
        print("-" * 50)

    print("\nRAG pipeline testing complete.")