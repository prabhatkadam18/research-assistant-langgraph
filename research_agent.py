from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from operator import add
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
import os
from langsmith import traceable

from dotenv import load_dotenv
load_dotenv()  


llm = ChatOllama (
    model="gemma4:12b",
    base_url=os.getenv("OLLAMA_BASE_URL") ,
    temperature=0
)

search_tool = TavilySearch(max_results=3)

class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    search_results: Annotated[list, add]   # append the new results in the previous ones
    iterations: int

class SubQuestions(BaseModel):
    questions: list[str] = Field(description="3 to 5 focused research sub-questions")
    
def planner(state: ResearchState) -> dict:
    prompt = f"Break this research topic into 3-5 focused sub-questions:\n\n{state['topic']}"
    result = llm.with_structured_output(SubQuestions).invoke(prompt)
    return {"sub_questions": result.questions}

def search_agent(state: ResearchState) -> dict:
    collected = []
    for q in state["sub_questions"]:
        hits = search_tool.invoke({ "query": q})
        collected.append({ "question": q, "results": hits})
        
    return {
        "search_results": collected,			# This is appended because of the add operator in the ResearchState type
        "iterations": state["iterations"] + 1	# This is replaced because of the default behavior replacing in state
	}

# NOTICE: how we have not added any @traceable decorator to any
# functions above. LangSmith still trace these. That is because of
# LANGSMITH_TRACING=true in env variables


# we create the graph here, creating the nodes and edges for the flow

graph = (
    StateGraph(ResearchState)
    .add_node("planner", planner)
    .add_node("search_agent", search_agent)
    .add_edge(START, "planner")
    .add_edge("planner", "search_agent")
    .add_edge("search_agent", END)
    .compile()
)

result = graph.invoke({
    "topic": "impact of remote work on software team productivity",
	"search_results": [],
    "iterations": 0
})

for item in result["search_results"]:
    print("Q: ", item["question"])
    print(" hits:", len(item["results"]))



