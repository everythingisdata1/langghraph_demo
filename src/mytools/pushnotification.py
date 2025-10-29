# Tool 2 - push
import os

import requests
from langchain_core.tools import Tool
from loguru import logger

PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_URL = os.getenv("PUSHOVER_URL")

logger.info("PushOver notification tools.....")


def push(txt: str):
    """ Send a push notification to the user """
    logger.info(txt)
    requests.post(PUSHOVER_URL, data={"token": PUSHOVER_USER, "user": PUSHOVER_TOKEN, "message": txt})


tool_push = Tool(
    name="send_push_notification",
    description="Useful when you want to send push notification.",
    func=push
)
