from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from ollama import MistralClient


def query_document(
    docs: list[Document],
    user_query: str,
    model_name: str,
    ollama_api_key: str,
    base_url: str,
    temperature: float = 0.3,
) -> str:
    """
    Queries the given documents using the provided parameters and returns the result.
    """
    # Define LLM chain
    llm = MistralClient(api_key=ollama_api_key)
    chain = get_map_reduce_chain(llm, user_query=user_query)

    # Invoke the chain on documents
    result = chain.invoke(docs)
    return result["output_text"]


def get_map_reduce_chain(llm: MistralClient, user_query: str) -> Chain:
    """
    Creates a map-reduce chain for querying documents.
    """
    # Map
    map_template = """The following is a set of documents:
    {docs}
    Based on this list of documents, please identify the information that is most relevant to the following query:
    {user_query}
    If the document is not relevant, please write "not relevant".
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template).partial(user_query=user_query)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is a set of partial answers to a user query:
    {docs}
    Take these and distill them into a final, consolidated answer to the following query:
    {user_query}
    Complete Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template).partial(user_query=user_query)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine chains
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
