import os
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from mistral_nemo import NemoClient  # Correct client for Nemo models
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
        # Retrieve the Mistral Nemo model
        llm = NemoClient(model_name=model_name)  # Use NemoClient for Mistral

        # Define the summarization prompt
        prompt_template = """Write a long summary of the following document. 
        Only include information that is part of the document. 
        Do not include your own opinion or analysis.

        Document:
        "{document}"
        Summary:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Set up the chain for LLM
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="document"
        )

        # Run the chain
        result = stuff_chain.invoke(docs)
        return result["output_text"]

    except Exception as e:
        return f"An unexpected error occurred: {e}"
