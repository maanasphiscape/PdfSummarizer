import streamlit as st
from documents_llm.summarize import summarize_document
from documents_llm.document import load_pdf
import os

# Streamlit App
st.title("Document Analyzer with Local Mistral-Nemo")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Save uploaded file temporarily
    with open("temp_uploaded_file.pdf", "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    # Load the PDF and extract content
    st.info("Loading the PDF...")
    docs = load_pdf("temp_uploaded_file.pdf")

    # Specify model details
    model_name = "mistral-nemo:latest"  # Update if needed
    base_url = "http://localhost:11434/v1"  # Ensure your local server is running

    # Summarize the document
    st.info("Summarizing the document...")
    try:
        summary = summarize_document(docs, model_name=model_name, base_url=base_url)
        # Display the summary
        st.subheader("Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")

    # Remove the temporary file
    os.remove("temp_uploaded_file.pdf")
