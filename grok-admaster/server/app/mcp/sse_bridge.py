"""
MCP SSE Bridge — Exposes stdio-based MCP servers over SSE transport.

n8n's MCP Client requires SSE (Server-Sent Events) transport. Our existing
MCP servers (mcp_cortex.py, mcp_dsp_server.py) use stdio transport.

This bridge runs both servers behind a single FastAPI app with SSE endpoints
that n8n (and any other MCP client) can connect to.

Usage:
    python -m app.mcp.sse_bridge              # Run on port 3001
    python -m app.mcp.sse_bridge --port 3002  # Custom port

Endpoints:
    GET  /sse           → SSE stream for MCP protocol messages
    POST /messages      → Send MCP requests (tool calls, etc.)
    GET  /health        → Health check
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# Ensure app imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger("mcp_sse_bridge")

app = FastAPI(title="Optimus Prime MCP SSE Bridge")

# ═══════════════════════════════════════════════════════════
#  Registry of all MCP tools from both servers
# ═══════════════════════════════════════════════════════════

_tools_registry: Dict[str, Dict[str, Any]] = {}
_initialized = False


async def _register_cortex_tools():
    """Register tools from mcp_cortex.py."""
    from app.mcp_cortex import (
        analyze_semantic_bleed,
        find_untapped_opportunities,
        get_semantic_health_report,
        embed_new_product,
        query_patrol_log,
    )

    _tools_registry["analyze_semantic_bleed"] = {
        "handler": analyze_semantic_bleed,
        "description": "Detect search terms wasting budget by being semantically distant from the target product.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asin": {"type": "string", "description": "The product ASIN to analyze"},
                "account_id": {"type": "integer", "description": "The client account ID"},
                "threshold": {"type": "number", "description": "Similarity threshold (0-1)", "default": 0.40},
            },
            "required": ["asin", "account_id"],
        },
    }
    _tools_registry["find_untapped_opportunities"] = {
        "handler": find_untapped_opportunities,
        "description": "Discover high-value search terms that are semantically related and converting but not targeted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asin": {"type": "string", "description": "The product ASIN"},
                "account_id": {"type": "integer", "description": "The client account ID"},
                "min_orders": {"type": "integer", "description": "Minimum orders to qualify", "default": 1},
            },
            "required": ["asin", "account_id"],
        },
    }
    _tools_registry["get_semantic_health_report"] = {
        "handler": get_semantic_health_report,
        "description": "Generate a semantic health report for an account (embeddings, patrols, bleeds).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "integer", "description": "The client account ID"},
            },
            "required": ["account_id"],
        },
    }
    _tools_registry["embed_new_product"] = {
        "handler": embed_new_product,
        "description": "Generate and store the semantic identity of a product ASIN.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asin": {"type": "string", "description": "The Amazon ASIN"},
                "title": {"type": "string", "description": "The product title"},
                "account_id": {"type": "integer", "description": "The client account ID"},
                "bullet_points": {"type": "array", "items": {"type": "string"}, "description": "Optional bullet points"},
            },
            "required": ["asin", "title", "account_id"],
        },
    }
    _tools_registry["query_patrol_log"] = {
        "handler": query_patrol_log,
        "description": "View recent autonomous patrol activity log.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of entries to return", "default": 20},
            },
        },
    }


async def _register_dsp_tools():
    """Register tools from mcp_dsp_server.py."""
    from app.mcp_dsp_server import (
        get_dsp_audiences,
        simulate_attack,
        analyze_market_position,
        get_ppc_recommendations,
        create_ppc_campaign,
    )

    _tools_registry["get_dsp_audiences"] = {
        "handler": get_dsp_audiences,
        "description": "Fetch active DSP audiences and their current metrics.",
        "inputSchema": {"type": "object", "properties": {}},
    }
    _tools_registry["simulate_attack"] = {
        "handler": simulate_attack,
        "description": "Simulate the impact of a DSP attack strategy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "budget": {"type": "number", "description": "Budget in USD"},
                "target_type": {"type": "string", "description": "Target type: conquest, retargeting, awareness"},
            },
            "required": ["budget", "target_type"],
        },
    }
    _tools_registry["analyze_market_position"] = {
        "handler": analyze_market_position,
        "description": "Analyze market position of an ASIN for a target keyword using live search data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Target keyword"},
                "asin": {"type": "string", "description": "The ASIN to analyze"},
            },
            "required": ["keyword", "asin"],
        },
    }
    _tools_registry["get_ppc_recommendations"] = {
        "handler": get_ppc_recommendations,
        "description": "Generate AI-powered PPC keyword and bid recommendations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asin": {"type": "string", "description": "Product ASIN"},
                "title": {"type": "string", "description": "Product title"},
                "strategic_context": {"type": "object", "description": "Optional context from market analysis"},
            },
        },
    }
    _tools_registry["create_ppc_campaign"] = {
        "handler": create_ppc_campaign,
        "description": "Provision a new PPC campaign in the Optimus Prime system.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Campaign name"},
                "daily_budget": {"type": "number", "description": "Daily budget in USD"},
                "strategy": {"type": "string", "description": "AI strategy: manual, auto_pilot, aggressive_growth, profit_guard"},
                "keywords": {"type": "array", "items": {"type": "object"}, "description": "Keyword list with text, match_type, bid"},
            },
            "required": ["name", "daily_budget", "strategy", "keywords"],
        },
    }


async def _ensure_initialized():
    global _initialized
    if not _initialized:
        await _register_cortex_tools()
        await _register_dsp_tools()
        _initialized = True
        logger.info(f"MCP SSE Bridge initialized with {len(_tools_registry)} tools")


# ═══════════════════════════════════════════════════════════
#  SSE Transport — MCP Protocol over Server-Sent Events
# ═══════════════════════════════════════════════════════════

# Active SSE sessions
_sessions: Dict[str, asyncio.Queue] = {}


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE stream endpoint. n8n connects here for MCP communication."""
    await _ensure_initialized()

    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _sessions[session_id] = queue

    logger.info(f"New SSE session: {session_id}")

    async def event_stream():
        # Send the endpoint message so client knows where to POST
        yield f"event: endpoint\ndata: /messages?session_id={session_id}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            _sessions.pop(session_id, None)
            logger.info(f"SSE session closed: {session_id}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/messages")
async def messages_endpoint(request: Request, session_id: str = ""):
    """Receive MCP JSON-RPC requests from n8n."""
    await _ensure_initialized()

    body = await request.json()
    queue = _sessions.get(session_id)
    if not queue:
        return Response(status_code=404, content="Session not found")

    method = body.get("method", "")
    msg_id = body.get("id")

    # Handle MCP protocol methods
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "optimus-prime-mcp",
                    "version": "1.0.0",
                },
            },
        }
        await queue.put(response)

    elif method == "notifications/initialized":
        # Client acknowledged — no response needed
        pass

    elif method == "tools/list":
        tools_list = []
        for name, tool_def in _tools_registry.items():
            tools_list.append({
                "name": name,
                "description": tool_def["description"],
                "inputSchema": tool_def["inputSchema"],
            })
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": tools_list},
        }
        await queue.put(response)

    elif method == "tools/call":
        tool_name = body.get("params", {}).get("name", "")
        arguments = body.get("params", {}).get("arguments", {})

        tool_def = _tools_registry.get(tool_name)
        if not tool_def:
            error_response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
            }
            await queue.put(error_response)
        else:
            try:
                handler = tool_def["handler"]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**arguments)
                else:
                    result = handler(**arguments)

                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result, default=str)}
                        ]
                    },
                }
                await queue.put(response)
            except Exception as e:
                logger.error(f"Tool execution error ({tool_name}): {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": str(e)},
                }
                await queue.put(error_response)

    elif method == "ping":
        await queue.put({"jsonrpc": "2.0", "id": msg_id, "result": {}})

    else:
        logger.warning(f"Unknown MCP method: {method}")
        await queue.put({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })

    return Response(status_code=202, content="Accepted")


@app.get("/health")
async def health():
    await _ensure_initialized()
    return {
        "status": "healthy",
        "service": "mcp-sse-bridge",
        "tools_registered": len(_tools_registry),
        "active_sessions": len(_sessions),
        "tools": list(_tools_registry.keys()),
    }


# ═══════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP SSE Bridge for Optimus Prime")
    parser.add_argument("--port", type=int, default=3001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting MCP SSE Bridge on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
