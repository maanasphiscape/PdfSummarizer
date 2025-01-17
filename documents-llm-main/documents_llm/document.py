from langchain_community.document_loaders.pdf import PyPDFLoader

def load_pdf(file):
    """Loads and extracts content from a PDF file."""
    loader = PyPDFLoader(file)
    return loader.load()
