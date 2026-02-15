import asyncio
import os
import sys
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# Ensure we can import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "app")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

async def run_autopilot(keyword: str, asin: str):
    print(f"\nAUTOPILOT ENGAGED")
    print(f"Objective: Optimize market position for '{keyword}' (ASIN: {asin})\n")
    
    # Initialize MCP Client
    mcp_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "app", "mcp_dsp_server.py"))
    
    mcp_client = MultiServerMCPClient(
        {
            "dsp": {
                "transport": "stdio",
                "command": "py", 
                "args": [mcp_server_path],
            }
        }
    )

    try:
        print("[1/4] Connecting to Neural Link (MCP Tools)...")
        tools = await mcp_client.get_tools()
        tool_map = {t.name: t for t in tools}
        
        # 1. Analyze Market
        print(f"[2/4] Analyzing Live Market Data for '{keyword}'...")
        analyze_tool = tool_map.get("analyze_market_position")
        if not analyze_tool:
            raise ValueError("Tool 'analyze_market_position' not found")
            
        analysis_result_raw = await analyze_tool.ainvoke({"keyword": keyword, "asin": asin})
        
        # Parse result: The MCP/LangChain adapter might return a string, a dict, or a list of content blocks
        analysis_result = {}
        if isinstance(analysis_result_raw, dict):
            analysis_result = analysis_result_raw
        elif isinstance(analysis_result_raw, str):
            try:
                analysis_result = json.loads(analysis_result_raw)
            except:
                 # It might be an error string
                 analysis_result = {"error": analysis_result_raw}
        elif isinstance(analysis_result_raw, list):
            # Likely a list of content blocks like [{'type': 'text', 'text': '...'}]
            try:
                text_content = analysis_result_raw[0].get("text", "{}")
                analysis_result = json.loads(text_content)
            except Exception as e:
                print(f"Failed to parse list output: {e}")
                analysis_result = {"raw": str(analysis_result_raw)}
                
        print(f"   > Market Analysis: {json.dumps(analysis_result, indent=2)}")
        
        # 2. Get Recommendations
        print(f"[3/4] Generating Strategy & Keywords...")
        rec_tool = tool_map.get("get_ppc_recommendations")
        if not rec_tool:
             raise ValueError("Tool 'get_ppc_recommendations' not found")
             
        # Pass the analysis result as strategic context
        rec_result_raw = await rec_tool.ainvoke({
            "asin": asin, 
            "title": keyword,  # Using keyword as title proxy for broad context if needed
            "strategic_context": analysis_result
        })
        
        rec_result = {}
        if isinstance(rec_result_raw, dict):
            rec_result = rec_result_raw
        elif isinstance(rec_result_raw, str):
            try:
                rec_result = json.loads(rec_result_raw)
            except:
                rec_result = {"error": rec_result_raw}
        elif isinstance(rec_result_raw, list):
             try:
                text_content = rec_result_raw[0].get("text", "{}")
                rec_result = json.loads(text_content)
             except:
                rec_result = {}
                
        print(f"   > Strategy Generated: {rec_result.get('strategy')} | Budget: ${rec_result.get('suggested_daily_budget')}")
        
        # 3. Provision Campaign
        print(f"[4/4] Provisioning Campaign...")
        create_tool = tool_map.get("create_ppc_campaign")
        if not create_tool:
             raise ValueError("Tool 'create_ppc_campaign' not found")
        
        strategy_name = rec_result.get("strategy", "manual")
        campaign_name = f"Autopilot_{keyword.replace(' ', '_')}_{strategy_name}"
        
        campaign_result = await create_tool.ainvoke({
            "name": campaign_name,
            "daily_budget": rec_result.get("suggested_daily_budget", 50.0),
            "strategy": strategy_name,
            "keywords": rec_result.get("recommended_keywords", [])
        })
        
        print("\n" + "="*50)
        print("          AUTOPILOT REPORT")
        print("="*50 + "\n")
        print(json.dumps(campaign_result, indent=2))
        print("\n" + "="*50)

    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path)
    
    # Test Parameters (Project Seed)
    target_asin = os.getenv("PROJECT_ASIN", "B0DWK3C1R7")
    target_keyword = "145W GaN Travel Charger"
    
    asyncio.run(run_autopilot(target_keyword, target_asin))
