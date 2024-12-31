import openai
import os  # To access environment variables
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import requests

# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("API key not found! Please set the OPENAI_API_KEY environment variable.")

# Optionally set the OpenAI API base URL globally
openai.api_key = openai_api_key  # Set the API key globally for OpenAI SDK
openai.api_base = "https://api.openai.com/v1"  # Optional if you want a custom base URL

def summarize_document(
    docs: list[Document],
    model_name: str,
    temperature: float = 0.1,
) -> str:
    try:
        # Initialize the LLM with the OpenAI API key set globally
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=openai_api_key,  # The API key is set globally, but you can still pass it if necessary
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
        return f"Connection error: {e}"

    except Exception as e:
        return f"An 
