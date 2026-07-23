from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import os

from dotenv import load_dotenv
load_dotenv()  

class State(TypedDict):
	name: str
	greeting: str


# @traceable
# def clean_name(state: State):
# 	return {"name": state['name'].strip().title()}

# @traceable
# def make_greeting(state: State):
# 	return {"greeting": f"Hello, {state['name']}! Welcome to LangGraph."}

# graph = (
# 	StateGraph(State)
# 	.add_node("clean_name", clean_name)
# 	.add_node("make_greeting", make_greeting)
# 	.add_edge(START, "clean_name")
# 	.add_edge("clean_name", "make_greeting")
# 	.add_edge("make_greeting", END)
# 	.compile()
# )

# result = graph.invoke({ "name": "Prabhat Kadam" })

# print(result)


llm = ChatOllama (
    model="gemma4:12b",
    base_url=os.getenv("OLLAMA_BASE_URL") ,
    temperature=0
)

@tool
def add(a: int, b: int):
	"Add two numbers"
	return a + b


resp = llm.bind_tools([add]).invoke("What is 2 + 3? Use the tool.")

print(resp.tool_calls)