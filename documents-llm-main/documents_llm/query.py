import os
import requests
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate

# Replace MistralClient with direct API call
def query_document(
    docs: list[Document],
    user_query: str,
    model_name: str,
    openai_api_key: str,
    base_url: str,
    temperature: float = 0.3,
) -> str:
    try:
        # API settings
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        payload = {
            "model": model_name,
            "temperature": temperature,
            "prompt": user_query,
            "documents": [doc.page_content for doc in docs]  # Assuming docs contain page content
        }

        # Call Ollama API directly
        response = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Error: {response.status_code}, {response.text}")
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate MapReduce chain
def get_map_reduce_chain(llm: LLMChain, user_query: str) -> Chain:
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of documents, please identify the information that is most relevant to the following query:
    {user_query} 
    If the document is not relevant, please write "not relevant"."""
    map_prompt = PromptTemplate.from_template(map_template)
    map_prompt = map_prompt.partial(user_query=user_query)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
    # Reduce
    reduce_template = """The following is set of partial answers to a user query:
    {docs}
    Take these and distill it into a final, consolidated answer to the following query:
    {user_query} 
    Complete Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_prompt = reduce_prompt.partial(user_query=user_query)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine and reduce documents
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    return map_reduce_chain
