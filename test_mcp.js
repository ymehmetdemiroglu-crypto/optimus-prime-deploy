const { spawn } = require('child_process');

const process = spawn('npx.cmd', [
    '-y', 'supergateway', '--streamableHttp',
    'https://sukunaita.app.n8n.cloud/mcp-server/http',
    '--header', 'authorization:Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjMGE1MjgwMi1hYjU5LTQ0NTQtYTMwYi03Yzg4NTFjODcwYjIiLCJpc3MiOiJuOG4iLCJhdWQiOiJtY3Atc2VydmVyLWFwaSIsImp0aSI6ImMyNjE2N2ZhLTJkODMtNDYzNC05ZjRjLTU1NzA0ZTg2Yzg3MyIsImlhdCI6MTc3MTY4MDkxNn0.2uiptlrl2NGjj0ktYJ_cN4kjGurecg316U4MDdKtWOk'
], { shell: true });

let buffer = '';

process.stdout.on('data', (data) => {
    buffer += data.toString();
    let msgs = buffer.split('\n');
    buffer = msgs.pop(); // keep remainder

    for (let m of msgs) {
        if (!m.trim()) continue;
        try {
            const msg = JSON.parse(m);
            console.log('RECV:', JSON.stringify(msg, null, 2));

            // If we got init response, list tools
            if (msg.id === 1 && !msg.method) {
                // Send init notification
                process.stdin.write(JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'notifications/initialized'
                }) + '\n');

                // List tools
                process.stdin.write(JSON.stringify({
                    jsonrpc: '2.0',
                    id: 2,
                    method: 'tools/list',
                    params: {}
                }) + '\n');
            } else if (msg.id === 2) {
                // got tools
                console.log('Got tools. Exiting.');
                process.kill();
                process.exit(0);
            }
        } catch (e) {
            console.log('Error parsing:', m);
        }
    }
});

process.stderr.on('data', (data) => {
    console.error('STDERR:', data.toString());
});

// Send init
const initReq = {
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: {
            name: 'test-client',
            version: '1.0.0'
        }
    }
};

process.stdin.write(JSON.stringify(initReq) + '\n');
