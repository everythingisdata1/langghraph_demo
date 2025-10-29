import asyncio
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from loguru import logger

def get_playwright_tools():
    # ✅ Ensure an event loop exists (Python 3.14 fix)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    logger.info("Creating async Playwright browser...")
    async_browser = create_async_playwright_browser(headless=True)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    return tools

if __name__ == "__main__":
    tools = get_playwright_tools()
    for tool in tools:
        logger.info(f"{tool.name} = {tool}")
