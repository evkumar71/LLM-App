from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pprint
import re

load_dotenv()


# Data cleaning function
def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


# load document and split it into chunks
document = TextLoader("./docs/dream.txt", encoding="utf8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(document)

print(f"Number of split documents: {len(split_docs)}")

# Clean the text in each document
texts = [clean_text(doc.page_content) for doc in split_docs]

# Load the OpenAI embeddings to vectorize the text
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# get a vectorstore from embeddings and texts
retriever = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 2})

# Query the retriever
query = "what did Martin Luther King Jr. dream about?"
docs = retriever.invoke(query)

# These are the relevant documents retrieved
print(f"Relevant docs: {docs}")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "you use the following docs {docs}, and answer the following question {query}"
)

model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(response)