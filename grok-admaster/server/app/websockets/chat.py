from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websockets.connection_manager import manager
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

from app.agents.grok_tools import GrokTools

# MCP client initialized once per process
_mcp_client = None


def _get_mcp_client():
    global _mcp_client
    if _mcp_client is None:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_dsp_server.py"))
        _mcp_client = MultiServerMCPClient(
            {
                "dsp": {
                    "transport": "stdio",
                    "command": "py",
                    "args": [app_path],
                }
            }
        )
    return _mcp_client


async def _create_agent_executor_once():
    """Create agent executor once per connection; reuse for all messages in that connection."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain import hub

    mcp_client = _get_mcp_client()
    mcp_tools = await mcp_client.get_tools()
    
    # Merge remote MCP tools with local RAG tools
    tools = mcp_tools + GrokTools.get_all_tools()
    llm = ChatOpenAI(
        model="openai/gpt-4o",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Optimus Pryme",
        },
        temperature=0,
    )
    try:
        prompt = hub.pull("hwchase17/openai-tools-agent")
    except Exception:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful Amazon PPC optimization assistant. Use tools when needed."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    return executor


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    executor = None
    try:
        executor = await _create_agent_executor_once()
    except Exception as e:
        await manager.send_personal_message(
            json.dumps({
                "id": "optimus-init-error",
                "sender": "grok",
                "content": f"Agent failed to initialize: {str(e)}. Check OPENROUTER_API_KEY and network.",
                "timestamp": datetime.now().isoformat(),
                "actions": [],
            }),
            websocket,
        )
    try:
        while True:
            data = await websocket.receive_text()
            if executor is None:
                response_content = "Agent not available. Please refresh and try again."
            else:
                try:
                    result = await executor.ainvoke({"input": data})
                    response_content = result.get("output", str(result))
                except Exception as e:
                    response_content = f"ERROR: Agent failed to process request. {str(e)}"

            response_data = {
                "id": f"optimus-{datetime.now().timestamp()}",
                "sender": "grok",
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "actions": [],
            }
            await manager.send_personal_message(json.dumps(response_data), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
