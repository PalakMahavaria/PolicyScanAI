import streamlit as st
import os
from rag_pipeline import get_rag_chain # Ensure this function exists in rag_pipeline.py

# --- Streamlit Page Configuration ---
# Set Streamlit page configuration for better layout and appearance
st.set_page_config(
    page_title="PolicyScan AI: Intelligent Policy Query",
    page_icon="🤖",
    layout="wide", # Use wide layout to utilize more screen space
    initial_sidebar_state="auto" # Sidebar automatically opens
)

# --- Custom CSS for basic styling ---
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6; /* Light gray background */
    color: #333333; /* Darker text for readability */
}
.st-emotion-cache-1kyxreq etr89bj1 { /* Target chat bubble for assistant */
    background-color: #e6f7ff; /* Light blue */
    border-radius: 15px;
    padding: 10px 15px;
}
.st-emotion-cache-h5g98r e1nzilvr1 { /* Target chat bubble for user */
    background-color: #dcf8c6; /* Light green */
    border-radius: 15px;
    padding: 10px 15px;
}
.stButton>button {
    border-radius: 10px;
    border: 1px solid #007bff;
    color: #007bff;
    background-color: white;
    padding: 8px 16px;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: #007bff;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State for Chat History ---
# This ensures chat messages persist across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Content ---
with st.sidebar:
    st.header("About PolicyScan AI 🤖")
    st.markdown("""
    This application is an **Intelligent Insurance Policy Query System**
    powered by **Google Gemini** and **LangChain**.

    It utilizes **Retrieval-Augmented Generation (RAG)** to provide
    accurate and cited answers directly from a knowledge base of
    insurance policy documents.

    **Key Features:**
    - Ask questions in natural language.
    - Get answers backed by specific policy documents.
    - Designed to streamline policy understanding for claims, underwriting, and customer service.

    ---
    **Developed by:** Palak Mahavaria
    """)
    st.markdown("[View Project on GitHub](https://github.com/PalakMahavaria/PolicyScanAI)")

    # Option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Rerun the app to clear the display


# --- Main Application Title and Introduction ---
st.title("🤖 PolicyScan AI: Intelligent Insurance Policy Query System")
st.markdown("""
Welcome! I'm your AI assistant designed to help you quickly find information
within complex insurance policy documents.

Simply ask a question about coverage, exclusions, definitions, or any other
policy-related detail, and I'll do my best to provide an accurate answer
sourced from the provided documents.
""")
st.info("💡 **Tip:** Try questions like 'What is the definition of Accident?' or 'Are pre-existing diseases covered?'")


# --- Load RAG Chain (without caching for now) ---
# This function loads your ChromaDB and sets up the RAG chain.
# We are intentionally NOT using @st.cache_resource here to avoid the previous RuntimeError.
def load_rag_chain():
    # get_rag_chain() is expected to handle the GOOGLE_API_KEY via os.getenv()
    # and load the ChromaDB from the 'chroma_db' directory.
    return get_rag_chain()

# Load the RAG chain when the app starts or reruns
# This is line 16 in app.py, where the previous error originated.
rag_chain = load_rag_chain()


# --- Display Chat Messages from History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Generation ---
prompt = st.chat_input("Ask a question about your policies...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Show a spinner while processing
            try:
                # Invoke the RAG chain with the user's prompt
                # The response will contain 'answer' and 'source_documents'
                response = rag_chain.invoke({"input": prompt})
                ai_answer = response["answer"]
                source_docs = response.get("source_documents", []) # Get source documents if available

                st.markdown(ai_answer)

                if source_docs:
                    st.markdown("---") # Separator for sources
                    st.markdown("**Sources:**")
                    # Display unique source file names
                    unique_sources = set()
                    for doc in source_docs:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            unique_sources.add(os.path.basename(doc.metadata['source']))
                    for source in sorted(list(unique_sources)):
                        st.markdown(f"- `{source}`")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

