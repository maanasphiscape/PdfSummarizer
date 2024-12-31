import os
import requests
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def summarize_document(
    docs: list[Document],
    model_name: str,
    base_url: str,
    temperature: float = 0.1,
) -> str:
    try:
        # Retrieve the OpenAI API key (or any key for Ollama)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("API key not found! Please ensure it is set in the .env file.")

        # Prepare the prompt template for summarization
        prompt_template = """Write a long summary of the following document. 
        Only include information that is part of the document. 
        Do not include your own opinion or analysis.

        Document:
        "{document}"
        Summary:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Make a request to the Ollama API (replace OpenAI API call)
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        payload = {
            "model": model_name,
            "temperature": temperature,
            "prompt": prompt.format(document=" ".join([doc.page_content for doc in docs])),
        }

        # Request the summarization from Ollama
        response = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers)

        if response.status_code == 200:
            # Return the summarized text from the response
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Error: {response.status_code}, {response.text}")

    except requests.exceptions.ConnectionError as e:
        return f"Connection error: {e}"

    except Exception as e:
        return f"An unexpected error occurred: {e}"
