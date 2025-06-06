from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="List the top 10 LLMs today"
)

print(response.output_text)
