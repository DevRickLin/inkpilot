import json
import subprocess
import shlex
import functools
import time
import asyncio # Added for running async SDK calls
from typing import List, Dict, Any, Optional, Union, Tuple, Type

# Pydantic imports for dynamic model creation
from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined

from langchain_core.tools import Tool

# Import necessary components from mcp-sdk
try:
    # Ensure you have 'pip install mcp-sdk'
    from mcp import ClientSession, StdioServerParameters, types as mcp_types
    from mcp.client.stdio import stdio_client
    MCP_SDK_AVAILABLE = True
except ImportError:
    print("Warning: mcp-sdk not found. Please install it ('pip install mcp-sdk') to use MCP features.")
    # Define dummy types/classes if SDK is not installed
    class StdioServerParameters: pass
    class ClientSession: pass
    class mcp_types:
        Tool = Dict # Dummy type
    async def stdio_client(*args, **kwargs): # Dummy async context manager
        class DummyClient:
            async def __aenter__(self): return (None, None) # Dummy reader/writer
            async def __aexit__(self, *args): pass
        return DummyClient()
    MCP_SDK_AVAILABLE = False

class ToolManager:
    """
    Manages connections to MCP servers (stdio and URL) and the tools they provide,
    using mcp-sdk for stdio communication where the SDK manages the process lifecycle.
    """
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        # Store only configuration and URL client placeholders
        self.server_configs: Dict[str, Dict] = {} # Raw config for each server
        self.url_clients: Dict[str, Any] = {} # Placeholder for URL clients
        self._prepare_server_configs() # Parses config, prepares URL clients
        self.system_prompt: Optional[str] = self._get_first_system_prompt() # Store the first found system prompt
        self.tools: List[Tool] = self._fetch_all_tools() # Fetch and process tools

    # No __del__ needed for stdio processes as SDK manages them per-call

    def _load_config(self) -> Dict:
        """Loads MCP server configuration from the JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                if "mcp_servers" not in config_data or not isinstance(config_data.get("mcp_servers"), list):
                    print(f"Warning: 'mcp_servers' key missing/invalid in '{self.config_path}'. No servers configured.")
                    config_data["mcp_servers"] = []
                return config_data
        except FileNotFoundError:
            print(f"Warning: Config file '{self.config_path}' not found.")
            return {"mcp_servers": []}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{self.config_path}': {e}")
            return {"mcp_servers": []}
        except Exception as e:
            print(f"Error loading config '{self.config_path}': {e}")
            return {"mcp_servers": []}

    def _prepare_server_configs(self) -> None:
        """Parses config, stores server info, and prepares URL client placeholders."""
        server_configs_list = self.config.get("mcp_servers", [])
        if not server_configs_list:
            print("No MCP servers found in configuration.")
            return

        print("Processing MCP server configurations...")
        for server_info in server_configs_list:
            server_name = server_info.get("name")
            server_type = server_info.get("type", "url")

            if not server_name:
                print(f"Warning: Skipping server config due to missing 'name': {server_info}")
                continue

            # Store raw config for later use by SDK or URL client
            self.server_configs[server_name] = server_info
            print(f"  Stored configuration for '{server_name}' (type: {server_type}).")

            # Prepare URL clients (if any) - STDIO processes are managed by SDK per-call
            if server_type == "url":
                mcp_url = server_info.get("url")
                if not mcp_url:
                    print(f"Warning: Skipping url server '{server_name}' (missing 'url').")
                    # Remove invalid config
                    if server_name in self.server_configs: del self.server_configs[server_name]
                    continue

                print(f"  Preparing placeholder for URL server '{server_name}' at {mcp_url}...")
                # In a real implementation, initialize your HTTP client here
                # self.url_clients[server_name] = ActualHttpClient(mcp_url, api_key=...)
                self.url_clients[server_name] = f"dummy_url_client_{server_name}" # Store placeholder

            elif server_type != "stdio":
                 print(f"Warning: Skipping server '{server_name}' (unsupported type: '{server_type}').")
                 if server_name in self.server_configs: del self.server_configs[server_name]


    def _create_stdio_params(self, server_name: str) -> Optional[StdioServerParameters]:
        """Helper to create StdioServerParameters from stored config."""
        server_info = self.server_configs.get(server_name)
        if not server_info or server_info.get("type") != "stdio":
             print(f"Error: Configuration for stdio server '{server_name}' not found or incorrect type.")
             return None

        command_config = server_info.get("command")
        if not command_config:
            print(f"Error: Missing 'command' in config for stdio server '{server_name}'.")
            return None

        cmd = ""
        args = []
        try:
            if isinstance(command_config, str):
                cmd_list = shlex.split(command_config)
                if cmd_list:
                    cmd = cmd_list[0]
                    args = cmd_list[1:]
            elif isinstance(command_config, list) and command_config:
                cmd = command_config[0]
                args = command_config[1:]
            else:
                 raise TypeError("Invalid command format.")

            if not cmd:
                 raise ValueError("Command cannot be empty.")

            # TODO: Add env handling from config if needed by StdioServerParameters
            return StdioServerParameters(command=cmd, args=args)

        except (TypeError, ValueError) as e:
             print(f"Error parsing command config for '{server_name}': {e}. Config: {command_config}")
             return None
        except Exception as e: # Catch potential shlex errors
            print(f"Unexpected error parsing command for '{server_name}': {e}")
            return None


    async def _async_fetch_stdio_tools(self, server_name: str) -> List[Dict]:
        """Async helper to fetch tools from a single stdio server using mcp-sdk."""
        if not MCP_SDK_AVAILABLE:
             print(f"Error: Cannot fetch tools from '{server_name}', mcp-sdk not available.")
             return []

        server_params = self._create_stdio_params(server_name)
        if not server_params:
             # Error already printed by helper
             return []

        tools = []
        session: Optional[ClientSession] = None
        try:
            print(f"  Attempting SDK connection to stdio server '{server_name}' for tool listing...")
            # SDK's stdio_client now manages the process lifecycle based on server_params
            async with stdio_client(server_params) as (reader, writer):
                if reader is None or writer is None:
                     # SDK likely failed to start/connect to the process
                     raise ConnectionError(f"SDK failed to establish stdio streams for '{server_name}'. Check server command/logs.")
                print(f"    SDK stdio streams established for '{server_name}'. Creating session...")
                # Create a client session within the connection context
                async with ClientSession(reader, writer) as session:
                    print(f"    SDK ClientSession created for '{server_name}'. Initializing...")
                    # Initialize the MCP connection
                    init_result = await session.initialize()
                    print(f"    MCP connection initialized for '{server_name}'. Server caps: {init_result.capabilities}")

                    # List the tools
                    print(f"    Listing tools for '{server_name}'...")
                    list_tools_result = await session.list_tools() # Get the result object
                    # Extract the actual list of tools (assuming attribute is 'tools')
                    fetched_tools_list: List[mcp_types.Tool] = list_tools_result.tools 
                    print(f"    Fetched {len(fetched_tools_list)} tool definitions from '{server_name}'.")
                    # Convert SDK Tool type to dictionary for consistent processing
                    # Ensure 'arguments' exists, defaulting to None or {} if needed
                    tools = [{"name": t.name,
                              "description": t.description,
                              "arguments": getattr(t, 'arguments', None)} # Safely get arguments
                             for t in fetched_tools_list] # Iterate over the extracted list
            print(f"  SDK connection to '{server_name}' closed after listing tools.")
            return tools
        except Exception as e:
            # Catch potential errors from stdio_client, initialize, list_tools etc.
            print(f"Error during SDK communication with '{server_name}' for tool fetch: {e}")
            # Log traceback for detailed debugging
            import traceback
            traceback.print_exc()
            # No process to clean up here as SDK context manager handles it
            # Config is kept so retry might be possible later? Or remove config:
            # if server_name in self.server_configs: del self.server_configs[server_name]
            return [] # Return empty list on error


    def _fetch_all_tools(self) -> List[Tool]:
        """Fetches tool definitions from all configured servers using appropriate methods."""
        all_raw_tools_data = []
        if not self.server_configs:
            print("Warning: No valid server configurations available. Cannot fetch tools.")
            return []

        print("\nFetching tools from configured servers...")
        server_names = list(self.server_configs.keys()) # Get a stable list of names

        servers_to_remove_config = [] # Track configs to remove after iteration

        for server_name in server_names:
            server_info = self.server_configs.get(server_name)
            if not server_info: continue # Should not happen, but safety check

            server_type = server_info.get("type", "url")
            raw_server_tools = []

            try:
                if server_type == "stdio":
                    # Run the async helper synchronously
                    raw_server_tools = asyncio.run(self._async_fetch_stdio_tools(server_name))
                    # If fetching failed, mark config for potential removal
                    if not raw_server_tools:
                         print(f"Marking stdio server '{server_name}' for config removal due to fetch failure.")
                         # servers_to_remove_config.append(server_name) # Option: remove failing configs

                elif server_type == "url":
                    # Placeholder logic for URL fetching
                    client_placeholder = self.url_clients.get(server_name)
                    if client_placeholder:
                        print(f"Placeholder: Fetching tools via HTTP for '{server_name}' - NOT IMPLEMENTED.")
                        # Replace with actual HTTP client call
                        # try:
                        #    raw_server_tools = self.url_clients[server_name].fetch_tools()
                        # except Exception as url_e: print(f"Error fetching URL tools.. {url_e}"); servers_to_remove_config.append(server_name)
                        raw_server_tools = [
                            {"name": f"{server_name}_url_tool1", "description": f"Placeholder Tool 1 from URL {server_name}"},
                            {"name": f"{server_name}_url_tool2", "description": f"Placeholder Tool 2 from URL {server_name}"}
                        ]
                        print(f"  Using {len(raw_server_tools)} placeholder tools for '{server_name}'.")
                    else:
                         print(f"Warning: No client placeholder found for URL server '{server_name}'.")
                         # servers_to_remove_config.append(server_name) # Option: remove if no client

                # Tag tools with server name and add to the main list
                for tool_info in raw_server_tools:
                    # Basic validation of tool structure
                    if isinstance(tool_info, dict) and 'name' in tool_info and 'description' in tool_info:
                        tool_info['server'] = server_name # Add context needed for processing
                        all_raw_tools_data.append(tool_info)
                    else:
                         print(f"Warning: Skipping invalid tool definition received from '{server_name}': {tool_info}")

            except Exception as e:
                 # Catch errors from asyncio.run or other issues during the loop for one server
                 print(f"Error processing server '{server_name}' during tool fetch loop: {e}")
                 # servers_to_remove_config.append(server_name) # Option: remove on any error

        # # Optional: Remove configurations for servers that failed persistently
        # for name in servers_to_remove_config:
        #      if name in self.server_configs:
        #          print(f"Removing configuration for server '{name}' due to persistent errors.")
        #          del self.server_configs[name]
        #      if name in self.url_clients:
        #           del self.url_clients[name] # Also clear URL client placeholder if config removed

        print(f"\nTotal raw tool definitions fetched: {len(all_raw_tools_data)}")
        processed_tools = self._process_mcp_tools(all_raw_tools_data)
        print(f"Processed {len(processed_tools)} tools into Langchain format.")
        return processed_tools


    def _process_mcp_tools(self, raw_tools_data: List[Dict]) -> List[Tool]:
        """Converts raw tool dictionaries into Langchain Tool objects."""
        processed = []
        print(f"\nProcessing {len(raw_tools_data)} raw tool definitions into Langchain Tools...")
        for tool_info in raw_tools_data:
            tool_name = tool_info.get('name')
            tool_desc = tool_info.get('description')
            server_name = tool_info.get('server') # Crucial context

            if not all([tool_name, tool_desc, server_name]):
                print(f"Warning: Skipping tool due to missing name/desc/server info: {tool_info}")
                continue

            # Create the execution wrapper using partial
            tool_func = functools.partial(self._execute_mcp_tool_wrapper,
                                          server_name=server_name,
                                          tool_name=tool_name)

            # --- Create args_schema from tool_info['arguments'] --- 
            args_schema: Optional[Type[BaseModel]] = None
            mcp_args_schema = tool_info.get('arguments')
            if mcp_args_schema and isinstance(mcp_args_schema, dict):
                try:
                    # Ensure tool_name is valid for a class name
                    safe_model_name = f"{tool_name}_ArgsModel"
                    args_schema = self._create_pydantic_model_from_schema(safe_model_name, mcp_args_schema)
                    print(f"  Successfully created args_schema for tool '{tool_name}'")
                except Exception as schema_e:
                    print(f"Warning: Could not create Pydantic args_schema for tool '{tool_name}' from provided schema. Error: {schema_e}. Schema: {mcp_args_schema}")
            # --- End args_schema creation ---

            try:
                # Create Langchain Tool, potentially with args_schema
                langchain_tool = Tool(
                    name=tool_name,
                    description=tool_desc,
                    func=tool_func,
                    args_schema=args_schema # Pass the generated schema
                )
                processed.append(langchain_tool)
                # print(f"  Created Langchain Tool: '{tool_name}' (Server: '{server_name}')") # Verbose
            except Exception as e:
                 print(f"Error creating Langchain Tool for '{tool_name}': {e}")

        return processed

    def _create_pydantic_model_from_schema(self, model_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        """Dynamically creates a Pydantic model from a JSON schema-like dictionary.

        Args:
            model_name: The desired name for the Pydantic model class.
            schema: A dictionary representing the JSON schema, typically from MCP tool arguments.
                Expected format: {'properties': { 'arg_name': {'type': 'string', 'description': '... '}}, 'required': ['arg_name']}

        Returns:
            A Pydantic model class.
        """
        fields = {}
        if not isinstance(schema, dict) or 'properties' not in schema or not isinstance(schema['properties'], dict):
            raise ValueError(f"Invalid schema format for '{model_name}'. Expected dict with 'properties' key.")

        properties = schema.get('properties', {})
        required_fields = schema.get('required', [])

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict, # Basic mapping, could be enhanced for nested objects
        }

        for name, prop_details in properties.items():
            if not isinstance(prop_details, dict) or 'type' not in prop_details:
                print(f"Warning: Skipping property '{name}' in '{model_name}' due to invalid format: {prop_details}")
                continue

            field_type_str = prop_details.get('type')
            field_type = type_mapping.get(field_type_str, Any) # Default to Any if type unknown

            # Handle array items type if specified
            if field_type is List and 'items' in prop_details and isinstance(prop_details['items'], dict):
                item_type_str = prop_details['items'].get('type')
                item_type = type_mapping.get(item_type_str)
                if item_type:
                    field_type = List[item_type] # type: ignore

            description = prop_details.get('description')

            # Determine if the field is required
            is_required = name in required_fields

            # Create Pydantic Field definition
            if is_required:
                # Required fields have ... (Ellipsis) as the default value
                field_definition = (field_type, Field(..., description=description))
            else:
                # Optional fields have a default value (use None for simplicity, could parse 'default' from schema)
                default_value = prop_details.get('default', None)
                # Pydantic V2 uses PydanticUndefined for optional fields without explicit defaults in Field
                if default_value is None:
                     field_definition = (Optional[field_type], Field(default=None, description=description)) # type: ignore
                else:
                     field_definition = (Optional[field_type], Field(default=default_value, description=description)) # type: ignore

            fields[name] = field_definition

        # Create the Pydantic model dynamically
        return create_model(model_name, **fields) # type: ignore

    async def _async_execute_stdio_tool(self, server_name: str, tool_name: str, tool_input: Union[str, dict]) -> str:
        """Async helper to execute a tool on a stdio server using mcp-sdk."""
        if not MCP_SDK_AVAILABLE:
            return f"Error: Cannot execute tool '{tool_name}', mcp-sdk not installed."

        server_params = self._create_stdio_params(server_name)
        if not server_params:
            return f"Error: Failed to create parameters for stdio server '{server_name}'."

        session: Optional[ClientSession] = None
        try:
            print(f"  Attempting SDK connection to stdio server '{server_name}' for tool execution...")
            # Establish connection and session for this specific call
            async with stdio_client(server_params) as (reader, writer):
                if reader is None or writer is None:
                     raise ConnectionError(f"SDK failed to establish stdio streams for '{server_name}' execution.")
                print(f"    SDK stdio streams established for '{server_name}'.")
                async with ClientSession(reader, writer) as session:
                    print(f"    SDK ClientSession created for '{server_name}'. Initializing...")
                    await session.initialize()
                    print(f"    MCP connection initialized. Calling tool '{tool_name}'...")

                    # --- Input Type Handling --- 
                    tool_args = tool_input
                    # Always try to parse if the input is a string
                    if isinstance(tool_input, str):
                        try:
                            parsed_input = json.loads(tool_input)
                            if isinstance(parsed_input, dict):
                                tool_args = parsed_input
                                print(f"    Parsed string input as JSON dictionary for tool '{tool_name}'.")
                            else:
                                # Raise error if JSON is valid but not a dict when a dict might be expected
                                raise ValueError(f"Tool '{tool_name}' received valid JSON, but expected a dictionary structure, received type {type(parsed_input).__name__}.")
                        except json.JSONDecodeError:
                            # If JSON parsing fails for the string input, wrap it in a dictionary.
                            print(f"    Input string for tool '{tool_name}' is not valid JSON. Wrapping in {{'input': ...}}.")
                            tool_args = {"input": tool_input}
                    # --- End Modification ---
                    # --- End Input Type Handling ---

                    print(f"    DEBUG: Calling '{tool_name}' with args type: {type(tool_args)}, value (partial): {str(tool_args)[:200]}")

                    # Call the tool via the SDK session
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    print(f"    Tool '{tool_name}' executed successfully on '{server_name}'.")
                    return json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)

        except ValueError as ve: # Catch specific validation errors raised above
             print(f"Input validation error for tool '{tool_name}' on '{server_name}': {ve}")
             raise # Re-raise ValueError to be caught by the wrapper
        except Exception as e:
            print(f"Error during SDK execution of tool '{tool_name}' on '{server_name}': {e}")
            import traceback
            traceback.print_exc()
            return f"Error executing tool '{tool_name}' on '{server_name}': {e}"


    def _execute_mcp_tool_wrapper(self, tool_input: Union[str, dict], server_name: str, tool_name: str) -> str:
        """Synchronous wrapper to execute MCP tools using the appropriate method."""
        print(f"\nExecuting tool '{tool_name}' on server '{server_name}'...")

        server_info = self.server_configs.get(server_name)
        if not server_info:
            return f"Error: Configuration for server '{server_name}' no longer available."

        server_type = server_info.get("type", "url")

        try:
            if server_type == "stdio":
                 try:
                      # Run the async execution helper synchronously
                      return asyncio.run(self._async_execute_stdio_tool(server_name, tool_name, tool_input))
                 except ValueError as ve: # Catch the specific ValueError from the async helper
                      # Return the validation error message directly to Langchain
                      return str(ve) 

            elif server_type == "url":
                 client_placeholder = self.url_clients.get(server_name)
                 if client_placeholder:
                      print(f"Placeholder: Would execute tool '{tool_name}' via HTTP on {server_name} - NOT IMPLEMENTED.")
                      return f"Placeholder execution result for {tool_name} on {server_name}."
                 else:
                      return f"Error: No client placeholder found for URL server '{server_name}'."
            else:
                 return f"Error: Unsupported server type '{server_type}' for tool execution."

        except Exception as e:
             # Catch errors from asyncio.run or other unexpected issues
             print(f"Unexpected error during wrapper execution for tool '{tool_name}' on '{server_name}': {e}")
             return f"Error: An unexpected error occurred while preparing to execute '{tool_name}'."


    def get_tools(self) -> List[Tool]:
        """Returns the aggregated list of processed Langchain tools."""
        if not self.tools:
             print("Warning: ToolManager has no successfully processed tools.")
        return self.tools

    def get_system_prompt(self) -> Optional[str]:
        """Returns the system prompt found in the configuration, if any."""
        return self.system_prompt

    def _get_first_system_prompt(self) -> Optional[str]:
        """Finds the first system prompt defined in the server configurations."""
        server_configs_list = self.config.get("mcp_servers", [])
        print("Searching for system prompt in server configurations...")
        for server_info in server_configs_list:
            prompt = server_info.get("system_prompt")
            if isinstance(prompt, str) and prompt:
                print(f"  Found system prompt in config for server: '{server_info.get('name')}'")
                return prompt
        print("  No system prompt found in any server configuration.")
        return None

    def shutdown_servers(self):
        """Shuts down any necessary components (e.g., closing persistent URL clients)."""
        # Stdio processes are managed by the SDK per-call, no Popen objects to track/kill here.
        print("\nShutting down ToolManager...")

        # Example: Close any persistent URL client connections if they exist
        # for name, client in self.url_clients.items():
        #    if hasattr(client, 'close'):
        #        print(f"  Closing connection for URL client '{name}'...")
        #        try:
        #             client.close()
        #        except Exception as e:
        #             print(f"    Error closing client '{name}': {e}")

        self.url_clients.clear()
        self.server_configs.clear()
        print("ToolManager shutdown complete.")

# Example Usage / Cleanup
if __name__ == "__main__":
    # Create dummy config for testing
    dummy_config = {
        "mcp_servers": [
            {
                "name": "dummy_stdio_server",
                "type": "stdio",
                # Replace with a simple script that implements basic JSON-RPC echo for testing
                # e.g., a python script reading/writing length-prefixed JSON
                "command": ["python", "path/to/dummy_mcp_stdio_server.py"] 
            },
            {
                 "name": "dummy_url_server",
                 "type": "url",
                 "url": "http://localhost:9999" # Dummy URL
            }
        ]
    }
    # Ensure the dummy server script path is correct or create one.
    config_filename = "temp_config.json"
    # Check if dummy server script exists before writing config?
    
    with open(config_filename, "w") as f:
        json.dump(dummy_config, f)

    manager = None
    try:
        manager = ToolManager(config_path=config_filename)
        available_tools = manager.get_tools()
        print("\nAvailable Langchain Tools:")
        for tool in available_tools:
            print(f"- Name: {tool.name}")
            print(f"  Description: {tool.description}")
            print(f"  Function: {tool.func}")
        
        # Example of invoking a tool (if a server was successfully connected)
        if available_tools:
             print("\nAttempting to run first tool...")
             try:
                 # Example input - adjust based on expected tool input
                 result = available_tools[0].run("test input") 
                 print(f"Tool Result: {result}")
             except Exception as e:
                  print(f"Error running tool: {e}")

    finally:
        if manager:
            manager.shutdown_servers() # Ensure servers are stopped
        # Clean up dummy config
        import os
        try:
            os.remove(config_filename)
            print(f"Removed {config_filename}")
        except FileNotFoundError:
             pass # Already removed or never created 