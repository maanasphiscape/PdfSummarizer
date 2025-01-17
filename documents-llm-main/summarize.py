from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Configure the local LLM
llm = ChatOpenAI(
    temperature=0.1,
    model_name="mistral-nemo",
    api_key="ollama",  # Placeholder API key
    base_url="http://localhost:11434/v1",  # Local server endpoint
)

# Prompt template for summarization
prompt_template = """Write a long summary of the following document. 
Only include information that is part of the document. 
Do not include your own opinion or analysis.

Document:
"{document}"
Summary:"""
prompt = PromptTemplate.from_template(prompt_template)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

def summarize_document(docs):
    """Summarizes the content of a document."""
    result = llm_chain.invoke(docs)
    return result["output_text"]
