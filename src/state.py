from typing import Annotated

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel

load_dotenv(verbose=True)


class State(BaseModel):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = ChatOpenAI(model="gpt-4o-mini")


def chatbot_node(old_state: State) -> State:
    logger.info(old_state)
    response = llm.invoke(old_state.messages)
    new_state = State(messages=[response])
    return new_state


graph_builder.add_node("chatbot", chatbot_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


def chat(user_input: str, history):
    initial_state = State(messages=[{"role": "user", "content": user_input}])
    result = graph.invoke(initial_state)
    print(result)
    return result['messages'][-1].content


gr.ChatInterface(chat, type="messages").launch()
