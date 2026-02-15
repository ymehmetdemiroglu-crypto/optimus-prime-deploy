import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from dotenv import load_dotenv

load_dotenv()

# Initialize Tavily Client
tavily_api_key = os.getenv("TAVILY_API_KEY")

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily."""
    if not tavily_api_key or tavily_api_key == "your_tavily_api_key_here":
        return "ERROR: TAVILY_API_KEY not configured. Research capability is limited."
    
    tavily_client = TavilyClient(api_key=tavily_api_key)
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

from app.services.dataforseo_client import dfs_client
from app.services.market_intelligence_ingester import market_ingester

async def amazon_search(query: str, location_code: int = 2840, persist: bool = True):
    """
    Search Amazon directly for products, prices, and reviews using DataForSEO.
    Results are automatically persisted to the database for long-term analysis.
    
    Args:
        query: The search term.
        location_code: Location code (default US: 2840).
        persist: Whether to save results to database (default True).
    """
    try:
        results = await dfs_client.get_amazon_products(query, location_code=location_code)
        
        # Persist to database if enabled
        if persist and results:
            try:
                await market_ingester.ingest_amazon_products(
                    keyword=query,
                    products=results,
                    mark_as_competitors=True
                )
            except Exception as db_err:
                # Log but don't fail the search if DB write fails
                import logging
                logging.warning(f"Failed to persist market data: {db_err}")
        
        # Simplify results for the agent
        simplified = []
        for item in results[:10]:
            simplified.append({
                "title": item.get("title"),
                "asin": item.get("asin"),
                "price": item.get("price"),
                "rating": item.get("rating"),
                "reviews": item.get("reviews_count"),
                "url": item.get("url")
            })
        return simplified
    except Exception as e:
        return f"Error calling DataForSEO: {str(e)}"

# Alias to maintain agent tool compatibility
searchapi_amazon_search = amazon_search 

async def amazon_product_details(asin: str):
    """
    Get information for a specific Amazon product using its ASIN via DataForSEO.
    (Note: DataForSEO uses the same search endpoint effectively for ASIN lookup if used as keyword, 
    or we could use the specific product endpoint if implemented. For now, we search by ASIN.)
    """
    try:
        results = await dfs_client.get_amazon_products(asin)
        if results:
            item = results[0]
            return {
                "title": item.get("title"),
                "asin": item.get("asin"),
                "price": item.get("price"),
                "rating": item.get("rating"),
                "reviews": item.get("reviews_count"),
                "url": item.get("url")
            }
        return "Product not found."
    except Exception as e:
        return f"Error calling DataForSEO Product: {str(e)}"

searchapi_amazon_product = amazon_product_details

research_instructions = """You are an expert researcher for Optimus Pryme, an Amazon Advertising command center.
Your job is to conduct thorough research on market trends, competitor strategies, and advertising best practices.

The core product you are optimizing for is B0DWK3C1R7 (Project Seed).

You have access to:
1. `internet_search`: Powered by Tavily for general web intel and news.
2. `searchapi_amazon_search`: Powered by SearchApi.io for direct Amazon product data (price, reviews, rank).
3. `searchapi_amazon_product`: Use this to get deep details on our product or specific competitor ASINs.

Be thorough, objective, and provide actionable insights for an Amazon seller.
"""

from langchain_openai import ChatOpenAI

def get_research_agent():
    """Create and return a Deep Research Agent configured with OpenRouter."""
    llm = ChatOpenAI(
        model="openai/gpt-4o",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Optimus Pryme Researcher",
        },
        temperature=0
    )
    
    return create_deep_agent(
        model=llm, # deepagents supports passing a custom model
        tools=[internet_search, searchapi_amazon_search, searchapi_amazon_product],
        system_prompt=research_instructions
    )

async def perform_research(query: str):
    """Convenience function to run research and return result."""
    agent = get_research_agent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
