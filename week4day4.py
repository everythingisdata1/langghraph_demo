from typing import TypedDict, Annotated

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger

from src.mytools.myplaywrite import get_playwright_tools
from src.mytools.pushnotification import tool_push

load_dotenv(override=True)
logger.info("Starting Week4Day4...")


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = get_playwright_tools()

all_tools = tools + [tool_push]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(all_tools)


def chatbot(state: State):
    logger.info(f"Chatbot called :: {state["messages"]}")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def graph_builder():
    logger.info("STEP1: Graph Builder ")
    builder = StateGraph(State)

    logger.info("STEP-2: Node")
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(tools))

    logger.info(f"STEP-3:: Graph edge")
    builder.add_conditional_edges("chatbot", tools_condition, "tools")
    builder.add_edge("tools", "chatbot")
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    logger.info("STEP-4: Graph buildr ")
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = graph_builder()

config = {"configurable": {"thread_id": "10"}}


async def chat(user_input: str, history):
    result = await graph.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    return result["messages"][-1].content


gr.ChatInterface(chat, type="messages").launch()
