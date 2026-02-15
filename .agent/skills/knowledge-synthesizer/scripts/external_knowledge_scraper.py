"""
External Knowledge Scraper
Fetches information from external sources like Blogs, GitHub, and News.
"""
import sys
import os
import asyncio
import json
import argparse
from typing import List, Dict, Optional
from datetime import datetime

# Path setup to find server module
current_dir = os.path.dirname(os.path.abspath(__file__))
# .agent/skills/knowledge-synthesizer/scripts -> optimus pryme
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
server_path = os.path.join(project_root, "grok-admaster", "server")
sys.path.append(server_path)

# Load env variables manually to ensure script works in isolation
from dotenv import load_dotenv
load_dotenv(os.path.join(server_path, '.env'))

try:
    from app.services.researcher import internet_search
except ImportError as e:
    print(f"Warning: Could not import researcher service: {e}")
    # Mock for testing if import fails
    def internet_search(query: str, max_results=5, topic="general"):
        return {"results": []}

class ExternalKnowledgeScraper:
    def __init__(self):
        pass

    def search_github_repos(self, topic: str, limit: int = 5) -> List[Dict]:
        """
        Research GitHub repositories related to a topic.
        Uses generic internet search targeted at github.com.
        """
        # Specific query to target GitHub repositories
        query = f"site:github.com {topic} repository"
        sys.stderr.write(f"Searching GitHub for: {topic}...\n")
        
        try:
            search_results = internet_search(query, max_results=limit)
        except Exception as e:
            sys.stderr.write(f"Data fetch error: {e}\n")
            return []
        
        results = []
        if isinstance(search_results, dict) and "results" in search_results:
            raw_results = search_results["results"]
            for res in raw_results:
                # Basic filter to ensure we get repo links usually
                if "github.com" in res.get("url", ""):
                    results.append({
                        "title": res.get("title"),
                        "url": res.get("url"),
                        "description": res.get("content", "")[:200] + "...",
                        "source": "github"
                    })
        elif isinstance(search_results, list): # Some Tavily versions return list directly
             for res in search_results:
                if "github.com" in res.get("url", ""):
                    results.append({
                        "title": res.get("title"),
                        "url": res.get("url"),
                        "description": res.get("content", "")[:200] + "...",
                        "source": "github"
                    })
        
        return results[:limit]

    def monitor_blogs(self, topic: str) -> List[Dict]:
        """Monitor industry blogs."""
        query = f"{topic} industry news blog analysis {datetime.now().year}"
        sys.stderr.write(f"Searching Blogs for: {topic}...\n")
        search_results = internet_search(query, max_results=5, topic="news")
        
        results = []
        if isinstance(search_results, dict) and "results" in search_results:
             results = search_results["results"]
        elif isinstance(search_results, list):
             results = search_results

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Knowledge Scraper")
    parser.add_argument("--source", choices=["github", "general"], default="general")
    parser.add_argument("--topic", required=True, help="Search topic")
    
    args = parser.parse_args()
    
    scraper = ExternalKnowledgeScraper()
    
    if args.source == "github":
        results = scraper.search_github_repos(args.topic)
        print(json.dumps(results, indent=2))
    else:
        results = scraper.monitor_blogs(args.topic)
        print(json.dumps(results, indent=2, default=str)) # Use default=str for datetime serialization if needed
