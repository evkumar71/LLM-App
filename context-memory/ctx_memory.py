from typing import Dict, List
from openai import OpenAI
import os, sys, json


def initialize_client(use_ollama: bool = False) -> OpenAI:

    if use_ollama:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_initial_messages() -> List[Dict[str, str]]:

    return [{"role": "system", "content": "You are a helpful assistant"}]


def chat(
    user_input: str, messages: List[Dict[str, str]], client: OpenAI, model_name: str
) -> str:

    # append message to save history
    messages.append({"role": "user", "content": user_input})

    try:
        # Get response using the API
        response = client.chat.completions.create(model=model_name, messages=messages)

        # Append assistant's response to the conversation
        ast_res = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ast_res})

        return ast_res

    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return "Sorry, I couldn't process your request right now."


def summarize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:

    # summarsize last five msg to reduce tokens and cost
    summary = "Previous conversation :" + " ".join(
        [m["content"][:50] for m in messages[-5:]]
    )

    return [{"role": "system", "content": summary}] + messages[-5:]


def save_conversation(
    messages: List[Dict[str, str]], filename: str = "conversation.json"
):

    # write conversation to a file
    with open(filename, "w") as f:
        json.dump(messages, f)


def load_conversation(filename: str = "conversation.json") -> List[Dict[str, str]]:

    try:
        with open(filename, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"No conversation file found at {filename}")
        return create_initial_messages()


def main():
    # Model selection
    print("Select model type:")
    print("1. OpenAI gpt-o4-mini")
    print("2. Ollama (Local)")

    choice = input("Enter choice (1 or 2): ")
    use_ollama = choice == "2"

    # Initialize client and model name
    client = initialize_client(use_ollama)
    model_name = "llama3.2" if use_ollama else "gpt-4o-mini"

    # Initialize or load conversation
    messages = create_initial_messages()

    print(f"\nUsing {'Ollama' if use_ollama else 'OpenAI'} model. Type 'quit' to exit.")
    print("Available commands:")
    print("- 'save': Save conversation")
    print("- 'load': Load conversation")
    print("- 'summary': Summarize conversation")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "save":
            save_conversation(messages)
            print("Conversation saved!")
            continue
        elif user_input.lower() == "load":
            messages = load_conversation()
            print("Conversation loaded!")
            continue
        elif user_input.lower() == "summary":
            messages = summarize_messages(messages)
            print("Conversation summarized!")
            continue

        response = chat(user_input, messages, client, model_name)
        print(f"\nAssistant: {response}")

        # Automatically summarize if conversation gets too long
        if len(messages) > 10:
            messages = summarize_messages(messages)
            print("\n(Conversation automatically summarized)")


# Example usage
if __name__ == "__main__":
    main()
