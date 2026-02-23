import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function main() {
    const transport = new StdioClientTransport({
        command: "npx.cmd",
        args: [
            "-y",
            "supergateway",
            "--streamableHttp",
            "https://sukunaita.app.n8n.cloud/mcp-server/http",
            "--header",
            "authorization:Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjMGE1MjgwMi1hYjU5LTQ0NTQtYTMwYi03Yzg4NTFjODcwYjIiLCJpc3MiOiJuOG4iLCJhdWQiOiJtY3Atc2VydmVyLWFwaSIsImp0aSI6ImMyNjE2N2ZhLTJkODMtNDYzNC05ZjRjLTU1NzA0ZTg2Yzg3MyIsImlhdCI6MTc3MTY4MDkxNn0.2uiptlrl2NGjj0ktYJ_cN4kjGurecg316U4MDdKtWOk"
        ]
    });

    const client = new Client({
        name: "test-client",
        version: "1.0.0"
    }, { capabilities: {} });

    await client.connect(transport);
    console.log("Connected to MCP server");

    const tools = await client.listTools();
    console.log("Tools available:", JSON.stringify(tools, null, 2));

    // We want to execute an orchestration workflow tool. Let's see what is available.
    process.exit(0);
}

main().catch(console.error);
