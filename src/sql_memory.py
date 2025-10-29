import sqlite3
from typing import TypedDict, Annotated

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger

from src.mytools.pushnotification import tool_push
from src.mytools.savertool import tool_search

load_dotenv(override=True)

logger.info("Start LangGraph with SQL lite in-memory")
# Tool - Web

tools = [tool_search, tool_push]

## SQL-Lite as memory Saver
logger.info(f"SqliteSaver added ....")
db_path = "langgraph.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory_saver = SqliteSaver(conn)

## STEP1- State Builder
logger.info("STEP-1- State Builder")


class State(TypedDict):
    messages: Annotated[list, add_messages]


##STEP2:
logger.info("STEP-2: Graph Builder")

graph_builder = StateGraph(State)

##STEP3: Setup lllm and bind tools
logger.info("STEP-3: Setup lllm and bind tools")

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    logger.info(state["messages"])
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

##STEP4: Add Edges
logger.info("STEP-4:  Add Edges")

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

## STEP5:
logger.info("STEP-5: Compile Graph with SQL Memory Saver check point")
graph = graph_builder.compile(checkpointer=sql_memory_saver)

confg = {"configurable": {"thread_id": "3"}}


def chat(user_input, history):
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=confg)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
