import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from show_graph import show_graph

load_dotenv()

# openai configuration
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_api_key, model=llm_name)


# Tavily configuration
tavily_api_key = os.getenv("TAVILY_API_KEY")
tool = TavilySearch(max_results=2)
tools = [tool]
model_with_tools = model.bind_tools(tools)


# rest = tool.invoke("What is the capital of France?")
# print(rest)


# This class is used to define the state of the graph.
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# This will be a node in the graph
def bot(state: State) -> dict:
    print(state["messages"])

    return {"messages": model_with_tools.invoke(state["messages"])}


# STEP#1 - build a graph
graph_builder = StateGraph(State)

# instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)  # Add the node to the graph


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# STEP#2 - add a start node
# The first argument is the unique node name
# The second argument is a function or object that will be called whenever
# the node is used.
graph_builder.add_node(node="bot", action=bot)

# STEP#3 - add an entry point in the graph
graph_builder.set_entry_point("bot")

# STEP#4 - add end point
# graph_builder.set_finish_point("bot")

# STEP#5 - compile the graph
graph = graph_builder.compile()

show_graph(graph)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"])
