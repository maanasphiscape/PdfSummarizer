from pathlib import Path
import streamlit as st
import logging
from PyPDF2 import PdfReader

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Fallback imports (adjust as needed)
try:
    from utils.document import load_pdf
except ImportError as e:
    logging.warning("'load_pdf' is not found; using built-in text extraction.")
    load_pdf = None

try:
    from utils.query import query_document
    from utils.summarize import summarize_document
except ImportError as e:
    logging.error("Error importing modules 'query' or 'summarize': %s", e)
    raise ImportError("Ensure the 'query.py' and 'summarize.py' modules exist.")


def save_uploaded_file(uploaded_file: "UploadedFile", output_dir: Path = Path("/tmp")) -> Path:
    """Saves an uploaded file to a temporary directory."""
    output_path = Path(output_dir) / uploaded_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def extract_text_with_pypdf(file_path: Path, start_page: int = None, end_page: int = None) -> str:
    """Extracts text from PDF using PyPDF2 as a fallback."""
    try:
        reader = PdfReader(file_path)
        pages = reader.pages[start_page:end_page] if start_page or end_page else reader.pages
        return "".join(page.extract_text() for page in pages)
    except Exception as e:
        raise ValueError(f"Error extracting text with PyPDF2: {e}")


def run_query(
    uploaded_file: "UploadedFile",
    summarize: bool,
    user_query: str,
    start_page: int,
    end_page: int,
    model_name: str,
    openai_api_key: str,
    openai_url: str,
    temperature: float,
) -> str:
    """
    Saves the uploaded file, processes it for text or summaries, and performs a query.
    """
    # Save the file to a temporary path
    st.write("Saving the uploaded file...")
    file_path = save_uploaded_file(uploaded_file, output_dir=Path("/tmp"))

    # Load the document using either custom logic or fallback
    st.write("Loading the document...")
    if load_pdf:
        docs = load_pdf(file_path, start_page=start_page, end_page=end_page)
    else:
        docs = extract_text_with_pypdf(file_path, start_page=start_page, end_page=end_page)

    # Clean up temporary file
    file_path.unlink()

    # Summarize or query the document
    try:
        if summarize:
            st.write("Summarizing the document...")
            return summarize_document(
                docs,
                model_name=model_name,
                openai_api_key=openai_api_key,
                base_url=openai_url,
                temperature=temperature,
            )
        st.write("Querying the document...")
        return query_document(
            docs,
            user_query=user_query,
            model_name=model_name,
            openai_api_key=openai_api_key,
            base_url=openai_url,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"Error during document processing: {e}")
