
import asyncio
import httpx
import sys
import os
from datetime import datetime

# Simple color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

BASE_URL = "http://localhost:8000/api/v1"

async def check_health(client):
    try:
        resp = await client.get("http://localhost:8000/health")
        if resp.status_code == 200:
            print(f"{Colors.GREEN}[✓] System Integrity: NORMAL{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}[X] System Integrity: CRITICAL (Status {resp.status_code}){Colors.ENDC}")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}[X] System Integrity: OFFLINE ({str(e)}){Colors.ENDC}")
        return False

async def get_dashboard_summary(client):
    try:
        resp = await client.get(f"{BASE_URL}/dashboard/summary")
        data = resp.json()
        print(f"\n{Colors.HEADER}=== COMMAND CENTER STATUS ==={Colors.ENDC}")
        print(f"{Colors.BOLD}Total Sales:{Colors.ENDC} ${data.get('total_sales', 'N/A')}")
        print(f"{Colors.BOLD}Total Spend:{Colors.ENDC} ${data.get('total_spend', 'N/A')}")
        print(f"{Colors.BOLD}ACoS:{Colors.ENDC} {data.get('acos', 'N/A')}%")
        print(f"{Colors.BOLD}ROAS:{Colors.ENDC} {data.get('roas', 'N/A')}")
        print("=============================\n")
    except Exception as e:
        print(f"{Colors.WARNING}Could not fetch dashboard summary: {str(e)}{Colors.ENDC}")

async def run_console():
    print(f"{Colors.CYAN}Initializing Grok AdMaster Operator Console...{Colors.ENDC}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        if await check_health(client):
            await get_dashboard_summary(client)
            print(f"{Colors.GREEN}Operator Mode Active. Ready for commands.{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}Operator Mode Failed to Initialize.{Colors.ENDC}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_console())
