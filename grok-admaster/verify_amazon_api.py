
import logging
import asyncio
import sys
import os
import json
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add server directory to path
sys.path.append(os.path.join(os.getcwd(), "server"))

from app.services.dataforseo_client import dfs_client

async def verify_amazon_data():
    keyword = "gaming keyboard"
    print(f"--- Verify DataForSEO Client (Polling Mode) for: '{keyword}' ---")
    
    # Creds (Hardcoded for verification if needed, or rely on .env if working)
    # Using the one that worked for auth check
    dfs_client.login = "240201919@st.biruni.edu.tr"
    dfs_client.password = "75b4e889aac15cd5"
    creds = f"{dfs_client.login}:{dfs_client.password}"
    encoded = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
    dfs_client._auth_header = f"Basic {encoded}"
    
    try:
        # Calls the updated client method which now has internal polling
        products = await dfs_client.get_amazon_products(keyword)
        
        if products:
            print(f"SUCCESS: Received {len(products)} Amazon products via Client.")
            print("\nSample Data:")
            print(json.dumps(products[0], indent=2))
        else:
            print("FAILURE: Client returned no products (List empty).")

    except Exception as e:
        print(f"ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_amazon_data())
