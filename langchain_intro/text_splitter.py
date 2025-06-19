from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20,
    length_function=len
    )

# load documents
docs = TextLoader("./docs/dream.txt", encoding="utf8").load()

# split documents into chunks
split_docs = text_splitter.split_documents(docs)

# Output the results
for i, split in enumerate(split_docs):
    print(f"Split {i+1}:\n{split}\n")