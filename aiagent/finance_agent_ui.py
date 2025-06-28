import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
from io import StringIO
from typing import TypedDict, List
from pydantic import BaseModel
from show_graph import show_graph
from langgraph.checkpoint.memory import MemorySaver
import functools
from cachetools import TTLCache

memory = MemorySaver()

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilySearch(max_results=2)

model_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=model_name, temperature=0.0)

cache = TTLCache(maxsize=100, ttl=300)  # Cache for 5 minutes

@functools.lru_cache(maxsize=128)
def cached_tavily_search(query: str):
    response = tavily.invoke(input=query)
    return response # ["results"]

class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


def gather_financials_node(state: AgentState):
    # Read the CSV file into a pandas DataFrame
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))

    # Convert the DataFrame to a string
    financial_data_str = df.to_string(index=False)

    # Combine the financial data string with the task
    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}


def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}


def research_competitors_node(state: AgentState):
    content = state["content"] or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor),
            ]
        )
        for q in queries.queries:
            # response = tavily.invoke(input=q)
            response = cached_tavily_search(q)

            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.invoke(input=q)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"


# start building the state graph
builder = StateGraph(AgentState)

# add all the nodes to the graph
builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)
builder.add_node("write_report", write_report_node)

# set the start node
builder.set_entry_point("gather_financials")

builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

# add all edges for the nodes
builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile(checkpointer=memory)

# show_graph(graph)

# ==== Streamlit UI ====
import streamlit as st


def main():
    st.title("Financial Performance Reporting Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
    )
    competitors = st.text_area("Enter competitor names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )

    if st.button("Start Analysis") and uploaded_file is not None:
        # Read the uploaded CSV file
        csv_data = uploaded_file.getvalue().decode("utf-8")

        # Initialize state with all required keys
        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
            # Add these initialized keys
            "content": [],
            "financial_data": "",
            "analysis": "",
            "competitor_data": "",
            "comparison": "",
            "feedback": "",
            "report": "",
        }

        state_placeholder = st.empty()

        config = {"configurable": {"thread_id": "1"}}

        try:
            final_state = None
            for s in graph.stream(initial_state, config):
                with state_placeholder.container():
                    st.write("Current State:", s)
                final_state = s

            if final_state and "report" in final_state:
                st.subheader("Final Report")
                st.markdown(final_state["report"])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

# ==== End Streamlit UI ====

# Exec this script with the command:
# streamlit run finance_agent_ui.py
