from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

def summarize_document(docs, model_name, base_url):
    """
    Summarizes a document using the specified LLM.
    
    Args:
        docs (list): List of document pages or content.
        model_name (str): Name of the locally hosted model.
        base_url (str): Base URL of the local model.

    Returns:
        str: Summary of the document.
    """
    # Define the prompt template
    prompt_template = """Write a long summary of the following document. 
    Only include information that is part of the document. 
    Do not include your own opinion or analysis.

    Document:
    "{document}"
    Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Set up the LLM with the local model
    llm = ChatOpenAI(
        temperature=0.1,
        model_name=model_name,
        api_key=None,  # No API key for local setup
        base_url=base_url,
    )

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Combine document pages into one string
    combined_docs = "\n".join(docs)

    # Run the LLM chain
    result = llm_chain.predict(document=combined_docs)
    return result
