import os
import csv
import pandas
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# get openai api-key
api_key = os.getenv("OPENAI_API_KEY")


class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type

        if model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, model_name="text-embedding-3-small"
            )

        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        elif model_type == "nomic":
            # using Ollama nomic-embed-text model
            # Exec this on terminal - ollama pull nomic-embed-text
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text",
            )


class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type

        if model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
        else:
            api_key = ("ollama",)
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            model_name = "ollama3.2"

    def generate_completions(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting model respoonse: {str(e)}"


def select_models():
    # Select LLM Model
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama2")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    # Select Embedding Model
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embeddings")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
        print("Please enter 1, 2, or 3")

    return llm_type, embedding_type
