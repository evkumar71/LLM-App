from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
                                    model="o4-mini",
                                    messages=[{'role': 'system', 'content': 'You are a helpful agent'},
                                            {'role': 'user', 'content' : 'Write a haiku poem on Google. Also, give a title to the poem'}]
                                          )

print(response.choices[0].message.content)