from create_csv import generate_csv
from simple_rag import *
import streamlit as st


def streamlit_app():
    st.set_page_config(page_title="Space Facts RAG", layout="wide")
    st.title("🚀 Space Facts RAG System")

    # Sidebar for model selection
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama Llama2",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["openai", "chroma", "nomic"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "chroma": "Chroma Default",
            "nomic": "Nomic Embed Text (Ollama)",
        }[x],
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.facts = generate_csv()

        # Initialize models
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        # Setup ChromaDB
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # If models changed, reinitialize
    if (
        st.session_state.llm_model.model_type != llm_type
        or st.session_state.embedding_model.model_type != embedding_type
    ):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )

    # Display available facts
    with st.expander("📚 Available Space Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact['fact']}")

    # Query input
    query = st.text_input(
        "Enter your question about space:",
        placeholder="e.g., What is the Hubble Space Telescope?",
    )

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Response")
                st.write(response)

            with col2:
                st.markdown("### 📖 References Used")
                for ref in references:
                    st.write(f"- {ref}")

            # Show technical details in expander
            with st.expander("🔍 Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")


if __name__ == "__main__":
    streamlit_app()
