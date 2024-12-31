import os
import logging
from pathlib import Path
from documents_llm.document import load_pdf
from documents_llm.query import query_document
from documents_llm.summarize import summarize_document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

def save_uploaded_file(uploaded_file: "UploadedFile", output_dir: Path = Path("/tmp")) -> Path:
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
) -> str:
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL", "http://localhost:5000/v1")

        if not openai_api_key:
            raise ValueError("Missing OpenAI API Key. Ensure it's set in the environment.")

        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file, output_dir=Path("/tmp"))
        
        # Load the document
        docs = load_pdf(file_path, start_page=start_page, end_page=end_page)
        file_path.unlink()

        if summarize:
            return summarize_document(
                docs, model_name=model_name, base_url=base_url, temperature=temperature
            )

        return query_document(
            docs, user_query=user_query, model_name=model_name, base_url=base_url, temperature=temperature
        )

    except ValueError as e:
        logging.error("Configuration Error: %s", e)
        return f"Configuration error: {e}"

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        return f"An unexpected error occurred: {e}"
