import os
from dotenv import load_dotenv

# Import the function to create the agent executor
from agents.main_agent import create_agent_executor

def run():
    # 1. Load environment variables (especially OpenAI API key)
    load_dotenv()

    # Ensure the OpenAI API key is set
    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")

    # 2. Create the agent executor
    agent_executor = create_agent_executor()

    # 3. Run the agent interaction loop
    print("Simple LangChain Agent started. Type 'exit' to quit.")
    chat_history = [] # Simple list for history in this example
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Invoke the agent executor
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        # Basic history management (display only for now)
        # A more robust solution would store structured messages
        print(f"Agent: {response['output']}")
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Agent: {response['output']}")

    print("Agent stopped.")

if __name__ == "__main__":
    run() 