import os

from langchain.agents import create_agent
from src.mascan.tools.fin_api import get_weekly_stock_prices
from src.mascan.tools.web_query import web_query
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT")

agent = create_agent(
    model=OPENAI_MODEL_DEFAULT,
    tools=[get_weekly_stock_prices],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Give me the current stock price of Apple Inc."}]}
)

if __name__ == "__main__":
    print(result["messages"][-1].content_blocks)
