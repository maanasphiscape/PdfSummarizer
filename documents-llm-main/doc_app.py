import os
import time
import streamlit as st
from dotenv import load_dotenv
from documents_llm.st_helpers import run_query

# Load environment variables
load_dotenv()

# Load model parameters
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-nemo:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000/v1")

st.title("Document Analyzer")
st.write(
    "This is a simple document analyzer that uses LLM models to summarize and answer questions about documents. "
    "You can upload a PDF or text file, and the model will summarize the document and answer questions about it."
)

with st.sidebar:
    st.header("Model")
    model_name = st.text_input("Model name", value=MODEL_NAME)
    temperature = st.slider("Temperature", value=0.1, min_value=0.0, max_value=1.0)

    st.header("Document")
    st.subheader("Upload a PDF file")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if file:
        st.write("File uploaded successfully!")

    st.subheader("Page range")
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("Start page:", value=0, min_value=0)
    with col2:
        end_page = st.number_input("End page:", value=-1)

    st.subheader("Query type")
    query_type = st.radio("Select the query type", ["Summarize", "Query"])

if query_type == "Query":
    user_query = st.text_area(
        "User query", value="What is the data used in this analysis?"
    )

if st.button("Run"):
    result = None
    start = time.time()
    if file is None:
        st.error("Please upload a file.")
    else:
        with st.status("Running...", expanded=True) as status:
            try:
                result = run_query(
                    uploaded_file=file,
                    summarize=query_type == "Summarize",
                    user_query=user_query if query_type == "Query" else "",
                    start_page=start_page,
                    end_page=end_page,
                    model_name=model_name,
                    temperature=temperature,
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
