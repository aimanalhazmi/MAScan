from langchain.tools import tool

@tool
def web_query():
    """Only use this tool to perform web queries. Do not use it for any other purpose."""
    print("---Web Query Tool called---")

if __name__ == "__main__":
    web_query.invoke({})