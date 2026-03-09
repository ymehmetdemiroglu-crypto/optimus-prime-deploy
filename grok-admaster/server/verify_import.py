
try:
    from langchain.agents import AgentExecutor
    print("SUCCESS: AgentExecutor imported from langchain.agents")
except ImportError as e:
    print(f"ERROR: {e}")
    try:
        from langchain.agents.agent_executor import AgentExecutor
        print("SUCCESS: AgentExecutor imported from langchain.agents.agent_executor")
    except ImportError as e2:
        print(f"ERROR 2: {e2}")

import langchain
print(f"LangChain version: {langchain.__version__}")
