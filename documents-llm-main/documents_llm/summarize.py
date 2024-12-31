from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def summarize_document(
    docs: list[Document],
    model_name: str,
    base_url: str,
    temperature: float = 0.1,
) -> str:
    try:
        # Retrieve the OpenAI API key from the environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("API key not found! Please ensure it is set in the .env file.")

        # Initialize the LLM with the OpenAI API key
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=openai_api_key,  # Pass API key explicitly
        )

        # Define the prompt template
        prompt_template = """Write a long summary of the following document. 
        Only include information that is part of the document. 
        Do not include your own opinion or analysis.

        Document:
        "{document}"
        Summary:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Create LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Combine the documents using StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="document"
        )

        # Run the chain and get the result
        result = stuff_chain.invoke(docs)
        return result["output_text"]

    except requests.exceptions.ConnectionError as e:
        return f"Connection error: {e}"  # Add a proper return statement

    except Exception as e:
        return f"An unexpected error occurred: {e}"  # Ensure exceptions are logged and returned
