from langchain.agents import create_agent
from tools.fin_api import finapi

agent = create_agent(
    model="openai:gpt-5.4",
    tools=[finapi],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

if __name__ == "__main__":
    print(result["messages"][-1].content_blocks)
