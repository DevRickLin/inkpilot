from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
# Assuming Langchain provides an adapter or you have a way to wrap MCP tools
# from langchain.tools import Tool # Example if needed for wrapping
# from langchain_community.adapters.mcp import create_mcp_tools # Fictional adapter example

# Import the ToolManager
from tools.tool_manager import ToolManager # Adjusted import path

def create_agent_executor() -> AgentExecutor:
    """Creates and returns the LangChain agent executor."""

    # 1. Initialize ToolManager and fetch tools
    tool_manager = ToolManager() # Assumes config.json is in the root or handles path correctly
    mcp_tools_raw = tool_manager.get_tools()

    # --------------------------------------------------------------------------
    # Placeholder: Adapt MCP tools for Langchain
    # You need to replace this section with the actual logic to convert
    # the raw tool data/functions from MCP into Langchain-compatible Tool objects.
    # This might involve using a specific adapter or manually wrapping each tool.
    # --------------------------------------------------------------------------
    langchain_tools = []
    # Example of manual wrapping (replace with your actual logic):
    # for tool_info in mcp_tools_raw:
    #     def tool_function(*args, **kwargs):
    #         # Logic to call the actual MCP tool using tool_manager.mcp_client
    #         print(f"Calling MCP tool: {tool_info['name']} with args: {args}, kwargs: {kwargs}")
    #         # Replace with actual call:
    #         # result = tool_manager.mcp_client.call_tool(tool_info['name'], *args, **kwargs)
    #         return f"Result from {tool_info['name']}" # Dummy result
    #
    #     langchain_tool = Tool(
    #         name=tool_info.get("name", "unknown_mcp_tool"),
    #         func=tool_function,
    #         description=tool_info.get("description", "An MCP tool.")
    #     )
    #     langchain_tools.append(langchain_tool)
    
    # Fictional adapter example (if available):
    # langchain_tools = create_mcp_tools(tool_manager.mcp_client, mcp_tools_raw) 

    print(f"Placeholder: Loaded {len(mcp_tools_raw)} raw tools from MCP. Adapt them for Langchain.")
    # Using raw tools directly for now - THIS WILL LIKELY FAIL without adaptation
    # Replace 'mcp_tools_raw' with 'langchain_tools' once adaptation is implemented
    tools = mcp_tools_raw # <<<< REPLACE THIS with langchain_tools when ready
    if not tools:
        print("Warning: No tools loaded from MCP. Agent might not function as expected.")

    # Add local tools if needed (e.g., hello_tool)
    # from src.tools.local.hello_tool import get_hello_world # Adjust import
    # tools.append(get_hello_world) # Add local tools to the list if desired
    
    # --------------------------------------------------------------------------

    # 2. Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini") # Or another model

    # 3. Create the Prompt Template
    # Get system prompt from ToolManager, fallback to default
    system_prompt_str = tool_manager.get_system_prompt() or "You are a helpful assistant that can use tools."
    print(f"Using System Prompt: {system_prompt_str[:100]}...") # Log the prompt being used (truncated)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_str),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 4. Create the Agent
    # Ensure the 'tools' list contains Langchain-compatible objects
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor 