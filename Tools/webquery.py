from langchain.tools import tool

@tool
def web_query():
    print("---Web Query Tool called---")