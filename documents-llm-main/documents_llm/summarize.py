import os
from ollama import MistralClient
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
    temperature: float = 0.1,
) -> str:
    try:
        # Initialize the Ollama model client
        llm = MistralClient()  # API key is fetched from the environment variable

        # Define the prompt template for summarization
        prompt_template = """Write a long summary of the following document. 
        Only include information that is part of the document. 
        Do not include your own opinion or analysis.

        Document:
        "{document}"
        Summary:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Create LLM chain with the provided prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Combine the documents using StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="document"
        )

        # Run the chain and get the result
        result = stuff_chain.invoke(docs)
        return result["output_text"]

    except Exception as e:
        return f"An unexpected error occurred: {e}"
