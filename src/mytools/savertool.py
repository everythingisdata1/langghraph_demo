from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from loguru import logger

logger.info("Call GoogleSerperAPIWrapper ...")
serper = GoogleSerperAPIWrapper()
serper.run("What is the capital of France?")
tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for when you need more information from an online search")
