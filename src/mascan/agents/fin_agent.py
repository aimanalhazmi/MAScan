import os

from langchain.agents import create_agent
from src.mascan.tools.fin_api import get_weekly_stock_prices
from src.mascan.tools.web_query import web_query
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT")

agent = create_agent(
    model=OPENAI_MODEL_DEFAULT,
    tools=[web_query, get_weekly_stock_prices],
    system_prompt="""
        You are a financial assistant. 
        Use the web_query tool to answer general questions and get_weekly_stock_prices to retrieve stock price data.
        Always provide sources for your information.
    """,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the current stock price of BMW and what is the capital of France?"}]}
)

if __name__ == "__main__":
    print(result["messages"][-1].content_blocks)
