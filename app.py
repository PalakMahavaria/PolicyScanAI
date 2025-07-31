import streamlit as st
from rag_pipeline import get_rag_chain
import os

# Set Streamlit page configuration
st.set_page_config(page_title="PolicyScan AI", page_icon="🤖")

# Load GOOGLE_API_KEY from .env (already handled by rag_pipeline, but good practice for app context)
# The actual API key check is in rag_pipeline.py, so if it fails there, app won't run.

# Initialize RAG chain (and cache it for performance)
@st.cache_resource
def load_rag_chain():
    return get_rag_chain()

rag_chain = load_rag_chain()

# --- Streamlit Application Interface ---
st.title("🤖 PolicyScan AI: Intelligent Insurance Policy Query System") # This is the main title of your app
st.markdown("Enter your questions about insurance policies below. The AI will provide answers based on the loaded policy documents and cite sources.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your policies..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Get AI response
        try:
            ai_response = rag_chain.invoke(prompt)
            response_content = ai_response

            # Display AI response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response_content)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

        except Exception as e:
            error_message = f"An error occurred: {e}. Please ensure your API key is correct and you have an active internet connection."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            # This is a test comment to force redeploy