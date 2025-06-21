import os
import re
from openai import OpenAI
import dotenv


openai_api_key = os.getenv("OPENAI_API_KEY")
dotenv.load_dotenv()

model_name = "gpt-3.5-turbo"
client = OpenAI(api_key=openai_api_key)


# read prompt from file
def read_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()


# create an an agent class
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=self.messages,
        )
        return response.choices[0].message.content


# These action functins are tools used by the agent to perform specific tasks.
# They can be called by the agent when needed.


def calculate(what):
    return eval(what)


def planet_mass(name):
    masses = {
        "Mercury": 0.33011,
        "Venus": 4.8675,
        "Earth": 5.972,
        "Mars": 0.64171,
        "Jupiter": 1898.19,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.413,
    }
    return f"{name} has a mass of {masses[name]} × 10^24 kg"


known_actions = {"calculate": calculate, "planet_mass": planet_mass}

prompt = read_prompt(file_path="./prompt.txt").strip()


# Regular expression to match action lines in the agent's response
action_re = re.compile(r"^Action: (\w+): (.*)$")


# Function to handle the interactive query
def query():
    bot = Agent(prompt)
    max_turns = int(input("Enter the maximum number of turns: "))
    i = 0

    while i < max_turns:
        i += 1
        question = input("You: ")
        result = bot(question)
        print("Bot:", result)

        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                print(f"Unknown action: {action}: {action_input}")
                continue
            print(f" -- running {action} {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
            result = bot(next_prompt)
            print("Bot:", result)
        else:
            print("No actions to run.")
            break


if __name__ == "__main__":
    query()
