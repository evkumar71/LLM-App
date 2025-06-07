from openai import OpenAI
from dotenv import load_dotenv
import os

my_model = 'gpt-4o-mini'

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#1 ==few-shot learning or prompting
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
print(completion.choices[0].message.content)

#2 == direct prompting
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'you are an assistang'},
                    {'role': 'user', 'content':"Write about climate change in 5 lines"}
                ]
            )

#3 == chain-of-thought prompt
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are a match tutor'},
                    {'role': 'user', 'content':"Solve this math problem step by step: \
                        If John has 5 apples and gives 2 to Mary, how many does he have left?"}
                ]
            )
print(completion.choices[0].message.content)

#4 == Instructional prompting
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are knowledgable personal trainer'},
                    {'role': 'user', 'content':"Write a 25-line eassay describing the benefits of exercise"}
                ]
            )
print(completion.choices[0].message.content)

#5 == Role-playing prompt
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are character in a fictional story'},
                    {'role': 'user', 'content':"Describe the setting"}
                ]
            )
print(completion.choices[0].message.content)

#6 == open-ended prompt
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are a philospher'},
                    {'role': 'user', 'content':"what is the meaning of life"}
                ]
            )
print(completion.choices[0].message.content)

#7 == temperature and top-p scaling
completion = client.chat.completions.create(
                model=my_model,
                messages= [
                    {'role': 'system', 'content': 'You are a creative writer'},
                    {'role': 'user', 'content':"what a creative tagline for a coffee shop"}
                ],
                # temperature=0.2,
                top_p=0.9
            )
print(completion.choices[0].message.content)