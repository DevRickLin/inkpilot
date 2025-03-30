from langchain_core.tools import tool

@tool
def get_hello_world() -> str:
    """Returns the classic 'hello world' string."""
    return "hello world" 