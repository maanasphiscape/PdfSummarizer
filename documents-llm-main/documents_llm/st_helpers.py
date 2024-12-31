from pathlib import Path
import streamlit as st
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

try:
    from .document import load_pdf
except ImportError as e:
    logging.error("Error importing 'load_pdf' from document: %s", e)
    raise

from .query import query_document
from .summarize import summarize_document


def save_uploaded_file(
    uploaded_file: "UploadedFile", output_dir: Path = Path("/tmp")
) -> Path:
    output_path = Path(output_dir) / uploaded_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def run_query(
    uploaded_file: "UploadedFile",
    summarize: bool,
    user_query: str,
    start_page: int,
    end_page: int,
    model_name: str,
    temperature: float,
    openai_api_key: str = None,  # Accept API key explicitly (optional)
) -> str:
    try:
        # Use the passed API key if provided, otherwise retrieve from environment
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL", "http://localhost:11434/v1")  # Default fallback

        if not openai_api_key:
            raise ValueError("Missing OpenAI API Key. Ensure it's set in the environment or passed as an argument.")

        st.write("Saving the uploaded file...")
        file_path = save_uploaded_file(uploaded_file, output_dir=Path("/tmp"))
        st.write("Loading the document...")
        docs = load_pdf(file_path, start_page=start_page, end_page=end_page)
        file_path.unlink()

        if summarize:
            st.write("Summarizing the document...")
            return summarize_document(
                docs,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
            )

        st.write("Querying the document...")
        return query_document(
            docs,
            user_query=user_query,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    except ValueError as e:
        logging.error("Configuration Error: %s", e)
        return f"Configuration error: {e}"

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        return f"An unexpected error occurred: {e}"
