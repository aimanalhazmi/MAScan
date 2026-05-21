from langchain.tools import tool

@tool
def finapi():
    """Only use this tool to perform financial API queries. Do not use it for any other purpose."""
    print("---FinAPI Tool called---")

if __name__ == "__main__":
    finapi.invoke({})
    