"""
This tool allows agents to perform web queries.
Inspired by Langchain's WebSearchTool.
@author: Tim
"""

import os

from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

@tool
def web_query():
    """Only use this tool to perform web queries. Do not use it for any other purpose."""
    print("---Web Query Tool called---")

if __name__ == "__main__":
    web_query.invoke({})