import os
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from mistral-nemo import NemoClient  # Correct client for Nemo models
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def query_document(
    docs: list[Document],
    user_query: str,
    model_name: str,
    base_url: str,
    temperature: float = 0.3,
) -> str:
    try:
        # Initialize Nemo client with the specified model name
        llm = NemoClient(model_name=model_name)  # Use NemoClient

        # Create the query chain
        chain = get_map_reduce_chain(llm, user_query=user_query)

        result = chain.invoke(docs)
        return result["output_text"]
    except Exception as e:
        return f"An error occurred: {e}"

def get_map_reduce_chain(llm: NemoClient, user_query: str) -> LLMChain:
    map_template = """The following is a set of documents
    {docs}
    Based on this list of documents, please identify the information that is most relevant to the following query:
    {user_query} 
    If the document is not relevant, please write "not relevant"."""
    map_prompt = PromptTemplate.from_template(map_template)
    map_prompt = map_prompt.partial(user_query=user_query)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """The following is a set of partial answers to a user query:
    {docs}
    Take these and distill it into a final, consolidated answer to the following query:
    {user_query}"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_prompt = reduce_prompt.partial(user_query=user_query)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    return LLMChain(llm=llm, prompt=combine_documents_chain)
