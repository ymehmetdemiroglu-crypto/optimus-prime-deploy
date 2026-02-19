from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websockets.connection_manager import manager
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

from app.agents.optimus_tools import OptimusTools

# MCP client initialized once per process
_mcp_client = None


def _get_mcp_client():
    global _mcp_client
    if _mcp_client is None:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_dsp_server.py"))
        # Use "python3" — the POSIX standard; "py" is Windows-only and will fail on Linux.
        _mcp_client = MultiServerMCPClient(
            {
                "dsp": {
                    "transport": "stdio",
                    "command": "python3",
                    "args": [app_path],
                }
            }
        )
    return _mcp_client


async def _create_agent_executor_once():
    """Create agent executor once per connection; reuse for all messages in that connection."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    mcp_client = _get_mcp_client()
    mcp_tools = await mcp_client.get_tools()

    # Merge remote MCP tools with local RAG tools
    tools = mcp_tools + OptimusTools.get_all_tools()
    llm = ChatOpenAI(
        model="openai/gpt-4o",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Optimus Prime",
        },
        temperature=0,
    )
    agent = create_react_agent(
        llm,
        tools,
        prompt="You are Optimus, the AI assistant for Optimus Prime — an Amazon PPC war room dashboard. Clients connect their Amazon Advertising APIs and you help them optimize campaigns, analyze performance, and manage ad spend. Use tools when needed.",
    )
    return agent


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    executor = None
    try:
        try:
            executor = await _create_agent_executor_once()
        except Exception as e:
            # Attempt to notify the client; ignore send errors (client may have left).
            try:
                await manager.send_personal_message(
                    json.dumps({
                        "id": "optimus-init-error",
                        "sender": "optimus",
                        "content": f"Agent failed to initialize: {str(e)}. Check OPENROUTER_API_KEY and network.",
                        "timestamp": datetime.now().isoformat(),
                        "actions": [],
                    }),
                    websocket,
                )
            except Exception:
                pass

        while True:
            data = await websocket.receive_text()
            if executor is None:
                response_content = "Agent not available. Please refresh and try again."
            else:
                try:
                    result = await executor.ainvoke({"messages": [{"role": "user", "content": data}]})
                    messages = result.get("messages", [])
                    response_content = messages[-1].content if messages else str(result)
                except Exception as e:
                    response_content = f"ERROR: Agent failed to process request. {str(e)}"

            response_data = {
                "id": f"optimus-{datetime.now().timestamp()}",
                "sender": "optimus",
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "actions": [],
            }
            await manager.send_personal_message(json.dumps(response_data), websocket)
    except WebSocketDisconnect:
        pass
    except Exception:
        # Catch-all: network resets, protocol errors, etc. — ensure cleanup still runs.
        pass
    finally:
        # Always remove from the active pool regardless of how we exit.
        manager.disconnect(websocket)
