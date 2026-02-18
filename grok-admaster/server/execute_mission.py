import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Add the server/app directory to sys.path
# Since this is in /server, we add current dir and current dir / app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

async def execute_mission(query: str):
    print(f"MISSION START: {query}\n")
    
    # Initialize MCP Client
    mcp_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "app", "mcp_dsp_server.py"))
    
    mcp_client = MultiServerMCPClient(
        {
            "dsp": {
                "transport": "stdio",
                "command": "py", # Use 'py' for Windows
                "args": [mcp_server_path],
            }
        }
    )

    try:
        print("[1/3] Fetching Tactical Tools...")
        tools = await mcp_client.get_tools()
        
        print("[2/3] Initializing Neural Link (GPT-4o via OpenRouter)...")
        llm = ChatOpenAI(
            model="openai/gpt-4o",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
            default_headers={
                "HTTP-Referer": "https://localhost:3000",
                "X-Title": "Optimus Prime Mission Control",
            },
            temperature=0
        )
        
        # Use create_agent as per documentation
        agent = create_agent(llm, tools)

        print("[3/3] Executing Operation...")
        # Note: input format for create_agent is usually messages
        response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        
        print("\n" + "="*50)
        print("          MISSION CONCLUDED - REPORT BELOW")
        print("="*50 + "\n")
        
        # Output is usually in response['messages'][-1].content
        if isinstance(response, dict) and "messages" in response:
            print(response["messages"][-1].content)
        elif isinstance(response, list):
            print(response[-1].content)
        else:
            print(response)
        print("\n" + "="*50)

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path)
    
    user_query = "Research high-intent keywords for a PPC campaign for the 'Stanley The Aerolight Transit Mug (Hammertone Green, 0.47L)'. Use Amazon search to find competitor trends and suggest a mix of exact and broad match keywords."
    asyncio.run(execute_mission(user_query))
