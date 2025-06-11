import chromadb

# vector db client
chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(collection_name)

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
query_text = "Hello World !"

# query the db and get results
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

print(results)