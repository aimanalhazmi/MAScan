import os

from langchain.agents import create_agent
from tools.fin_api import finapi
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT")

agent = create_agent(
    model=OPENAI_MODEL_DEFAULT,
    tools=[finapi],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Give me the current stock price of Apple Inc."}]}
)

if __name__ == "__main__":
    print(result["messages"][-1].content_blocks)
