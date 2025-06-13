import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

# vector db client
chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(collection_name, embedding_function=default_ef)

# Define text documents to be added to vector-db
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
]

# load documents in vector db
for doc in documents:
    collection.upsert(ids=doc['id'], documents=doc['text'])
    
# query to search in db
query_text = "Age of the earth"

# query the db and get results
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

# format and print results
line_1 = '-' * 10
line_2 = '-' * 20

print(f'{line_1} Results {line_1}')
print(results)

print(f'{line_1} Query {line_1}')
print(query_text)

print(f'{line_1} Detailed {line_1}')
for idx, doc in enumerate(results['documents'][0]):
    doc_id = results['ids'][0][idx]
    dist = results['distances'][0][idx]
    
    print(f'Similar doc: <{doc}> -> (ID: {doc_id}, Distnace: {dist})')