from langchain_community.document_loaders import (
    # DirectoryLoader,
    # TextLoader,
    PyPDFLoader,
    # CSVLoader,
)

pdf_loader = PyPDFLoader("./docs/linux-manual.pdf")

pdf_docs = pdf_loader.load()

print(pdf_docs)