from ollama import MistralClient
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate

def query_document(
    docs: list[Document],
    user_query: str,
    model_name: str,
    base_url: str,
    temperature: float = 0.3,
) -> str:
    try:
        # Initialize the Ollama model client
        llm = MistralClient()  # API key is fetched from the environment variable

        # Define the map and reduce chains
        chain = get_map_reduce_chain(llm, user_query=user_query)

        # Run the chain and get the result
        result = chain.invoke(docs)
        return result["output_text"]

    except Exception as e:
        return f"An error occurred: {e}"

def get_map_reduce_chain(llm: MistralClient, user_query: str) -> Chain:
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of documents, please identify the information that is most relevant to the following query:
    {user_query} 
    If the document is not relevant, please write "not relevant". 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_prompt = map_prompt.partial(user_query=user_query)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
    # Reduce
    reduce_template = """The following is a set of partial answers to a user query:
    {docs}
    Take these and distill it into a final, consolidated answer to the following query:
    {user_query} 
    Complete Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_prompt = reduce_prompt.partial(user_query=user_query)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combine documents
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Final map-reduce chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    return map_reduce_chain
