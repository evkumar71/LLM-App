from chromadb.utils import embedding_functions

# create a default embedding function
emb_fn = embedding_functions.DefaultEmbeddingFunction()

# sample text to be converted to embedding
sample_text = 'Hello'

# convert text to embedding
emb = emb_fn(sample_text)

print(emb)