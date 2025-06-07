from openai import OpenAI
from dotenv import load_dotenv
import os

my_model = 'gpt-4o-mini'

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# client = OpenAI()

# ==few-shot learning or prompting
completion = client.chat.completions.create(
                model=my_model,
                messages=[
                    {'role':'system', 'content': 'You are a translator'},
                    {'role' :'user',
                     'content': """Translate these sentences: 
                                    'Hello' -> 'Hola', 
                                    'Goodbye' -> 'Adiós'. 
                                    '.
                                    Now translate: 'Thank you'."""
                    }
                    ]
                )

# print(completion.choices[0].message.content)
# == direct prompting
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'you are an assistang'},
                    {'role': 'user', 'content':"Write about climate change in 5 lines"}
                ]
            )

print(completion.choices[0].message.content)