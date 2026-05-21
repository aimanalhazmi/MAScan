"""
This tool allows agents to perform web queries.
Inspired by Langchain's WebSearchTool.
@author: Tim
"""

import os
import requests

from langchain.tools import tool
from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_QUERY_LIMIT = 3

firecrawl_client = Firecrawl(api_key=FIRECRAWL_API_KEY)

@tool
def web_query(
    query: str,
) -> str:
    """
    Perform a web query using the FireCrawl API. Only use this tool for web queries.
    Args:
        query (str): The search query to perform.
    Returns:
        str: The search results from the FireCrawl API.
    """

    results = firecrawl_client.search(
        query,
        limit=FIRECRAWL_QUERY_LIMIT,
        scrape_options={
            "formats": ["markdown", "links"],
            "blockAds": True
        }
        )
    
    return results

if __name__ == "__main__":

    results = web_query.invoke({"query": "What is the capital of France?"})
    print(results)