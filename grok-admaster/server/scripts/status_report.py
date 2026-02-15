
import httpx
import asyncio
import json

async def main():
    try:
        async with httpx.AsyncClient() as client:
            print("Connecting to Grok AdMaster (Headless Mode)...")
            
            # Fetch Summary
            try:
                summary = await client.get("http://127.0.0.1:8000/api/v1/dashboard/summary")
                summary_data = summary.json()
                print("\n=== SYSTEM STATUS REPORT ===")
                print(f"Sales: ${summary_data.get('total_sales', 'N/A')}")
                print(f"Spend: ${summary_data.get('total_spend', 'N/A')}")
                print(f"ACoS: {summary_data.get('acos', 'N/A')}%")
                print(f"ROAS: {summary_data.get('roas', 'N/A')}")
            except Exception as e:
                print(f"Failed to fetch summary: {e}")

            # Fetch Health Check
            try:
                # Assuming /health exists based on docs
                health = await client.get("http://127.0.0.1:8000/health")
                print(f"\nSystem Health: {health.status_code}")
            except:
                print("\nSystem Health: UNKNOWN")

            print("\nReady for command input.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
