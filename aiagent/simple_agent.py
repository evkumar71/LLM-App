import os
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


agent = Agent(system=prompt)


response = agent("What is the combined mass of Saturn and Jupiter and Mars?")
print(response)

next_prompt = "Observation: {}".format(planet_mass("Saturn"))

response = agent(next_prompt)
print(response)

next_prompt = "Observation: {}".format(planet_mass("Jupiter"))

response = agent(next_prompt)
print(response)

next_prompt = "Observation: {}".format(planet_mass("Mars"))

response = agent(next_prompt)
print(response)

next_prompt = "Observattion: {}".format(eval("568.34 + 1898.19"))

response = agent(next_prompt)

print(response)

