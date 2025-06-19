from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic} ")

model = ChatOpenAI(model="gpt-4o-mini")

# chaining is smiliar to pipes in shell scripting
chain = prompt | model | StrOutputParser() 

response = chain.invoke({"topic": "cats"})
print(response)