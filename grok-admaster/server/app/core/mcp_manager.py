import asyncio
import subprocess
import logging
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class MCPProcessManager:
    """Robust MCP subprocess manager with health monitoring and auto-restart."""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self.last_heartbeat = None
        self.restart_count = 0
        self.max_restarts = 5
        self.logger = logging.getLogger("mcp_manager")
        self._monitor_task = None
        
    async def start(self):
        """Start MCP server subprocess."""
        if self.process and self.process.poll() is None:
            self.logger.warning("MCP server already running")
            return
            
        self.logger.info(f"Starting MCP server: {self.server_script}")
        
        # Ensure we use the correct python executable
        python_exe = os.sys.executable
        
        try:
            self.process = subprocess.Popen(
                [python_exe, self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.last_heartbeat = datetime.utcnow()
            
            # Start health monitor if not already running
            if not self._monitor_task or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._health_monitor())
                
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            raise
        
    async def _health_monitor(self):
        """Monitor subprocess health and restart if needed."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            if not self.is_alive():
                self.logger.error("MCP server process found dead, restarting...")
                await self.restart()
                
            # Check heartbeat timeout
            if self.last_heartbeat:
                elapsed = datetime.utcnow() - self.last_heartbeat
                if elapsed > timedelta(minutes=5):
                    self.logger.error("MCP server heartbeat timeout (5 mins), forcing restart...")
                    await self.restart()
                    
    def is_alive(self) -> bool:
        """Check if subprocess is running."""
        return self.process is not None and self.process.poll() is None
        
    async def restart(self):
        """Restart MCP server with backoff."""
        if self.restart_count >= self.max_restarts:
            self.logger.critical("Max restart attempts reached for MCP server. Manual intervention required.")
            # Reset counter after a long delay or keep failing? 
            # For now, we stop trying to avoid infinite loops
            return
            
        self.logger.info(f"Restarting MCP server (Attempt {self.restart_count + 1})...")
        await self.stop()
        
        # Exponential backoff
        wait_time = min(60, 2 ** self.restart_count)
        await asyncio.sleep(wait_time)
        
        try:
            await self.start()
            self.restart_count += 1
        except Exception:
            self.logger.error("Restart failed.")
        
    async def stop(self):
        """Gracefully stop MCP server."""
        if self.process:
            self.logger.info("Stopping MCP server...")
            try:
                self.process.terminate()
                # Wait for process to terminate
                for _ in range(5):
                    if self.process.poll() is not None:
                        break
                    await asyncio.sleep(1)
                
                if self.process.poll() is None:
                    self.logger.warning("MCP server refused to terminate, killing...")
                    self.process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping MCP server: {e}")
            finally:
                self.process = None
                
    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to MCP server with timeout."""
        if not self.is_alive():
            await self.start()
            
        try:
            # Send command via stdin
            cmd_json = json.dumps(command) + "
"
            self.process.stdin.write(cmd_json)
            self.process.stdin.flush()
            
            # Read response with timeout
            # Note: subprocess.stdout.readline is blocking. 
            # In a production async environment, we'd use a non-blocking stream.
            # This is a simplified version.
            loop = asyncio.get_event_loop()
            line = await asyncio.wait_for(
                loop.run_in_executor(None, self.process.stdout.readline),
                timeout=30.0
            )
            
            if not line:
                raise RuntimeError("Empty response from MCP server")
                
            response = json.loads(line)
            self.last_heartbeat = datetime.utcnow()
            return response
            
        except asyncio.TimeoutError:
            self.logger.error("MCP server command timeout.")
            await self.restart()
            raise
        except Exception as e:
            self.logger.error(f"Error communicating with MCP server: {e}")
            await self.restart()
            raise
