
try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
    from langchain.agents import AgentExecutor
    print("AgentExecutor found in langchain.agents")
except ImportError as e:
    print(f"ImportError: {e}")
    try:
        from langchain.agents.agent_executor import AgentExecutor
        print("AgentExecutor found in langchain.agents.agent_executor")
    except ImportError as e2:
        print(f"ImportError 2: {e2}")

import sys
print(sys.path)
