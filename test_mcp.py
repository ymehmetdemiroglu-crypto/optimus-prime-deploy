import asyncio
import json
import subprocess
import sys

async def run_mcp():
    process = await asyncio.create_subprocess_exec(
        "npx.cmd", "-y", "supergateway", "--streamableHttp", 
        "https://sukunaita.app.n8n.cloud/mcp-server/http", 
        "--header", "authorization:Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjMGE1MjgwMi1hYjU5LTQ0NTQtYTMwYi03Yzg4NTFjODcwYjIiLCJpc3MiOiJuOG4iLCJhdWQiOiJtY3Atc2VydmVyLWFwaSIsImp0aSI6ImMyNjE2N2ZhLTJkODMtNDYzNC05ZjRjLTU1NzA0ZTg2Yzg3MyIsImlhdCI6MTc3MTY4MDkxNn0.2uiptlrl2NGjj0ktYJ_cN4kjGurecg316U4MDdKtWOk",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    process.stdin.write((json.dumps(init_req) + "\n").encode())
    await process.stdin.drain()
    
    # Read init response
    line = await process.stdout.readline()
    print("INIT MSG:", line.decode().strip())
    
    # Send initialized notification
    process.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
    await process.stdin.drain()
    
    # List tools
    tools_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    process.stdin.write((json.dumps(tools_req) + "\n").encode())
    await process.stdin.drain()
    
    line = await process.stdout.readline()
    print("TOOLS:", line.decode().strip())
    
    process.terminate()

if __name__ == "__main__":
    asyncio.run(run_mcp())
