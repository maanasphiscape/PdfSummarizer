import os
import time

import streamlit as st
from dotenv import load_dotenv

from documents_llm.st_helpers import run_query

# Load environment variables
load_dotenv()

# Load model parameters from environment
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")  # Default fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = os.getenv("OPENAI_URL", "http://localhost:11434/v1")  # Default fallback

# Application title and description
st.title("Document Analyzer")
st.write(
    "This is a simple document analyzer that uses LLM models to summarize and answer questions about documents. "
    "You can upload a PDF or text file, and the model will summarize the document and answer questions about it."
)

# Sidebar for configuration
with st.sidebar:
    st.header("Model Configuration")

    # Model settings
    model_name = st.text_input("Model name", value=MODEL_NAME)
    temperature = st.slider("Temperature", value=0.1, min_value=0.0, max_value=1.0)

    # File upload
    st.header("Document Upload")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file:
        st.success("File uploaded successfully!")

    # Page range inputs
    st.subheader("Page Range")
    start_page = st.number_input("Start page:", value=0, min_value=0)
    end_page = st.number_input("End page:", value=-1)

    # Query type selection
    st.subheader("Query Type")
    query_type = st.radio("Select the query type", ["Summarize", "Query"])

# Query input (only shown for "Query" type)
user_query = ""
if query_type == "Query":
    user_query = st.text_area("User Query", value="What is the data used in this analysis?")

# Run button
if st.button("Run"):
    if not file:
        st.error("Please upload a file.")
    else:
        result = None
        start = time.time()

        with st.spinner("Processing..."):
            try:
                # Run the query or summarization
                result = run_query(
                    uploaded_file=file,
                    summarize=query_type == "Summarize",
                    user_query=user_query if query_type == "Query" else "",
                    start_page=start_page,
                    end_page=end_page,
                    model_name=model_name,
                    openai_api_key=OPENAI_API_KEY,
                    temperature=temperature,
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                result = ""

        # Display the result
        if result:
            st.header("Result")
            st.markdown(result)
            st.info(f"Time taken: {time.time() - start:.2f} seconds ⏱️")

