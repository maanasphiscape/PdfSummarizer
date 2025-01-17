import streamlit as st
from documents_llm.summarize import summarize_document
from documents_llm.document import load_pdf

# Streamlit App
st.title("Document Analyzer with Local Mistral-Nemo")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Load the PDF and extract content
    st.info("Loading the PDF...")
    docs = load_pdf(uploaded_file)

    # Summarize the document
    st.info("Summarizing the document...")
    summary = summarize_document(docs)

    # Display the summary
    st.subheader("Summary")
    st.write(summary)
