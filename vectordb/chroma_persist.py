import chromadb
from chromadb.utils import embedding_functions

# default embedding fn
default_ef = embedding_functions.DefaultEmbeddingFunction()

# client to persist data
client = chromadb.PersistentClient(path='../db/chroma_persist')

# collection
collection = client.get_or_create_collection( \
                "my_story", embedding_function=default_ef)

# docs to be added to db
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
]

# load into db
for doc in documents:
    collection.upsert(ids=doc['id'], documents=doc['text'])
    
# sample query to be run on db
query_text = "Age of earth"

# search and get results
results = collection.query(query_texts=[query_text], 
                           n_results=2,
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
