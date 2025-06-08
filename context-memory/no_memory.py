from openai import OpenAI
import os
import sys

def simple_chat_without_memory(user_input:str,
                               use_ollama: bool=False) -> str:
    """
    This function demonstrates a chatbot WITHOUT memory/context management.
    Each call is independent and has no knowledge of previous interactions.
    """
    # Initialize OpenAI API (or Ollama)
    if use_ollama:
        client = OpenAI(base_url="http://localhost:11434", api_key="ollama")
        model_name = "llama3.2"
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model_name = 'gpt-4o-mini'
        
    # Each call only includes the current message
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        # Extract the assistant's reply
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return "Sorry, I couldn't process your request right now."

def main():
    # Model selection
    print("\n=== Simple Chatbot WITHOUT Memory ===")
    print("Notice how the bot won't remember anything from previous messages!")
    print("\nSelect model type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama (Local)")

    while True:
        choice = input("Enter choice 1 or 2: ").strip()
        
        if choice in ("1", "2"):
            break
        
    use_ollama = choice == 2
    
    # Print instructions
    print("\n=== Chat Session Started ===")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the screen")
    print("Each message is independent - the bot has no memory of previous messages!\n")
    
    while True:
        user_input = input("\nYou:")
        
        if not user_input:
            break
        
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye")
            sys.exit()
            
        # clear screen
        if user_input.lower() == 'clear':
            print("\033[H\033[J", end="")
            continue
                    
        # get the response
        res = simple_chat_without_memory(user_input, use_ollama)
        print(f"\nBot: {res}")
  
        # Visual separator for better readability
        print("\n" + "-" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat session ended by user. Goodbye! 👋")
        sys.exit()