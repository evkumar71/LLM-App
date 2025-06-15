import os
from chromadb.utils import embedding_functions
from openai import OpenAI
import chromadb

# openai config
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# get the embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small",
)

# chromadb config
chroma_client = chromadb.PersistentClient(path="../db/chroma_persist")

# collection to be written to db
collection = chroma_client.get_or_create_collection(
    name="document_qa_collection",
    embedding_function=openai_ef,
)


# fn to load docs in a list
def load_documents(dir_path):
    print("Loading docs from disk")
    documents = []
    for fn in os.listdir(dir_path):
        if fn.endswith(".txt"):
            fil_path = os.path.join(dir_path, fn)

            # context manager
            with open(fil_path, "r", encoding="utf-8") as file:
                documents.append({"id": fn, "text": file.read()})

    return documents


# fn to split text to chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks


# fn to chunk all docs
def chunk_docs(documents):
    print("splitting docs to chunks")
    chunked_docs = []
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_docs.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    return chunked_docs


# fn to generate openai embeddings
def get_openai_embedding(text):
    response = openai_client.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    emb = response.data[0].embedding
    return emb


# fn to write embeddings and doc to chromadb
def load_embedding_to_db(chunked_docs):
    print("gen embedding and write to db")
    for doc in chunked_docs:
        doc["embedding"] = get_openai_embedding(doc["text"])
        collection.upsert(
            ids=[doc["id"]], embeddings=[doc["embedding"]], documents=[doc["text"]]
        )


# query embeddings in db
def query_documents(question, n_results=2):
    # serach by embidding
    query_emb = get_openai_embedding(question)
    results = collection.query(query_embeddings=query_emb, n_results=n_results)

    # serach by query-text
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]

    return relevant_chunks


# pass the user-question along with context to get llm-response
def generate_llm_response(question, relevant_chunks):
    context = relevant_chunks

    prompt = """
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n {} \n\nQuestion:\n {}"
        """.format(
        context, question
    )

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful agent"},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message


# main start
doc_location = "../data/news_articles"
documents = load_documents(doc_location)
print(f"Len of loaded docs: {len(documents)}")

chunked_docs = chunk_docs(documents)

load_embedding_to_db(chunked_docs)

# main section, complete workflow
user_question = "Tell me about AI replacing TV writers strike"
relevant_chunks = query_documents(user_question)
answer = generate_llm_response(user_question, relevant_chunks)

print("---user Question---")
print(user_question)
print("---LLM response---")
print(answer.content)
