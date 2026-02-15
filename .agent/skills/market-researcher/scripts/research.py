import argparse
import sys
import os
import json
import asyncio

# Add server directory to path so we can import app modules
# Assuming we run this from project root
# Try multiple strategies to find server
current_dir = os.path.dirname(os.path.abspath(__file__))
# Strategy specific to .agent/skills location
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
server_path = os.path.join(project_root, "grok-admaster", "server")
sys.path.append(server_path)

try:
    from app.services.researcher import internet_search, searchapi_amazon_search, searchapi_amazon_product, perform_research
except ImportError:
    # Fallback to direct path search if above fails (e.g. running from root)
    if os.path.exists("grok-admaster/server"):
         sys.path.append(os.path.abspath("grok-admaster/server"))
    try:
        from app.services.researcher import internet_search, searchapi_amazon_search, searchapi_amazon_product, perform_research
    except ImportError as e:
        print(f"Error importing researcher service: {e}")
        sys.exit(1)

async def run_action(args):
    result = None

    if args.action == "internet_search":
        # Tavily is sync
        result = internet_search(args.query)
    elif args.action == "amazon_search":
        # DataForSEO is async
        result = await searchapi_amazon_search(args.query) # Domain arg ignored by DataForSEO (defaults to US/2840)
    elif args.action == "product_details":
        # DataForSEO is async
        result = await searchapi_amazon_product(args.query)
    elif args.action == "deep_research":
        # Deep Research is async
        result = await perform_research(args.query)

    print(json.dumps(result, indent=2, default=str))

def main():
    parser = argparse.ArgumentParser(description="Market Research Tool")
    parser.add_argument("action", choices=["internet_search", "amazon_search", "product_details", "deep_research"], help="The research action to perform")
    parser.add_argument("--query", "-q", help="Search query or ASIN")
    parser.add_argument("--domain", "-d", default="amazon.com", help="Amazon domain (ignored for DataForSEO)")
    
    args = parser.parse_args()
    
    # Run async loop
    asyncio.run(run_action(args))

if __name__ == "__main__":
    main()
