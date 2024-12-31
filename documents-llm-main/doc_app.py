import os
import time
import streamlit as st
from dotenv import load_dotenv
from documents_llm.st_helpers import run_query

# Load environment variables
load_dotenv()

# Debugging: print the loaded Ollama API key
ollama_api_key = os.getenv("OLLAMA_API_KEY")
if not ollama_api_key:
    st.error("Ollama API Key is not set. Please check the .env file or Streamlit secrets.")
else:
    st.write("Ollama API Key loaded successfully.")

# Load model parameters
MODEL_NAME = os.getenv("MODEL_NAME", "mixtral:latest")
OLLAMA_API_KEY = ollama_api_key
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

st.title("Document Analyzer (Powered by Mistral)")
st.write(
    "This is a document analyzer using the Mistral model from Ollama. "
    "Upload a PDF or text file, and the model will summarize and answer questions about it."
)

with st.sidebar:
    st.header("Model Configuration")

    model_name = st.text_input("Model name", value=MODEL_NAME)
    temperature = st.slider("Temperature", value=0.1, min_value=0.0, max_value=1.0)

    st.header("Document Upload")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file:
        st.write("File uploaded successfully!")

    st.subheader("Page Range")
    st.write(
        "Select page range. Pages are numbered starting at 0. Negative values count from the end, e.g., -1 is the last page."
    )
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("Start page:", value=0, min_value=0)
    with col2:
        end_page = st.number_input("End page:", value=-1)

    st.subheader("Query Type")
    query_type = st.radio("Select the query type", ["Summarize", "Query"])

if query_type == "Query":
    user_query = st.text_area("Enter your question", value="What is the data used in this analysis?")

if st.button("Run"):
    result = None
    start = time.time()
    if file is None:
        st.error("Please upload a file.")
    else:
        with st.status("Running...", expanded=True) as status:
            try:
                # Pass Ollama parameters to the query
                result = run_query(
                    uploaded_file=file,
                    summarize=query_type == "Summarize",
                    user_query=user_query if query_type == "Query" else "",
                    start_page=start_page,
                    end_page=end_page,
                    model_name=model_name,
                    temperature=temperature,
                    ollama_api_key=ollama_api_key,
                )
                status.update(label="Done!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Error", state="error", expanded=False)
                st.error(f"An error occurred: {e}")
                result = ""

        if result:
            with st.container(border=True):
                st.header("Result")
                st.markdown(result)
                st.info(f"Time taken: {time.time() - start:.2f} seconds", icon="⏱️")
