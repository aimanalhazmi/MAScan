from langchain.tools import tool

@tool
def finapi():
    print("---FinAPI Tool called---")