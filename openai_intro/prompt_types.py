from openai import OpenAI
from dotenv import load_dotenv
import os

my_model = 'gpt-4o-mini'

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

# == chain-of-thought prompt
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are a match tutor'},
                    {'role': 'user', 'content':"Solve this math problem step by step: \
                        If John has 5 apples and gives 2 to Mary, how many does he have left?"}
                ]
            )
print(completion.choices[0].message.content)

# == Instructional prompting
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are knowledgable personal trainer'},
                    {'role': 'user', 'content':"Write a 25-line eassay describing the benefits of exercise"}
                ]
            )
print(completion.choices[0].message.content)