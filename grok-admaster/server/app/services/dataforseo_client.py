
import httpx
import base64
import logging
from typing import Optional, Dict, List, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataForSEOClient:
    """
    Client for DataForSEO API.
    Handles authentication and requests for Amazon and SEO data.
    """
    BASE_URL = "https://api.dataforseo.com/v3"

    def __init__(self):
        self.login = settings.DATAFORSEO_LOGIN
        self.password = settings.DATAFORSEO_PASSWORD
        self._auth_header = None
        
        if self.login and self.password:
            creds = f"{self.login}:{self.password}"
            encoded = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
            self._auth_header = f"Basic {encoded}"
        else:
            logger.warning("DataForSEO credentials not configured.")

    async def _post_request(self, endpoint: str, payload: List[Dict]) -> Optional[Dict]:
        """Generic POST request handler."""
        if not self._auth_header:
            logger.error("Attempted DataForSEO request without credentials.")
            return {"error": "Missing Credentials"}

        url = f"{self.BASE_URL}{endpoint}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": self._auth_header, "Content-Type": "application/json"},
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"DataForSEO Request Failed: {str(e)}")
                return {"error": str(e)}

    async def get_amazon_products(self, keyword: str, location_code: int = 2840, language_code: str = "en") -> List[Dict]:
        """
        Search for products on Amazon.
        Default location: US (2840).
        """
        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "depth": 10
        }]
        
        # Using the Amazon SERP Organic Task Post (Queue) + Polling
        # Live endpoint is not available for this specific API part
        post_response = await self._post_request("/serp/amazon/organic/task_post", payload)
        
        if not post_response or "tasks" not in post_response:
             logger.error("Failed to queue Amazon SERP task")
             return []
             
        task_id = post_response["tasks"][0]["id"]
        
        # Poll for results (Max 60 seconds)
        import asyncio
        for i in range(30):
            await asyncio.sleep(2)
            get_response = await self._post_request(f"/serp/amazon/organic/task_get/advanced/{task_id}", [])
            
            if get_response and "tasks" in get_response:
                tasks = get_response["tasks"]
                if not tasks:
                    continue
                    
                task = tasks[0]
                # Log status every 5 checks to avoid spam, or if it's not 20000/40602
                if i % 5 == 0:
                     logger.info(f"Task {task_id} status: {task['status_code']} ({task['status_message']})")

                if task["status_code"] == 20000:
                    # Success
                    data = get_response
                    break
                elif task["status_code"] == 40602:
                    # Creating/Processing
                    continue
                else:
                    logger.error(f"Task failed: {task['status_message']}")
                    return []
        else:
             logger.error("Timeout waiting for Amazon SERP results")
             return []
        
        if not data or "tasks" not in data:
            return []

        # Parse results
        results = []
        tasks = data.get("tasks")
        
        # Debugging: Print raw data if needed or just handle None
        if tasks is None:
            logger.warning(f"DataForSEO response 'tasks' is None. Raw data: {data}")
            return []
            
        for task in tasks:
            if task.get("result"):
                for item in task["result"][0].get("items", []):
                    results.append({
                        "asin": item.get("asin"),
                        "title": item.get("title"),
                        "price": item.get("price", {}).get("value"),
                        "rating": item.get("rating", {}).get("value"),
                        "reviews_count": item.get("rating", {}).get("votes_count"),
                        "rank": item.get("rank_group"),
                        "url": item.get("url")
                    })
        return results

    async def get_keyword_volume(self, keyword: str, location_code: int = 2840) -> Dict[str, Any]:
        """
        Get search volume and CPC data for a keyword (Google data as proxy usually).
        """
        payload = [{
            "keywords": [keyword],
            "location_code": location_code,
            "language_code": "en"
        }]
        
        data = await self._post_request("/keywords_data/google_ads/search_volume/live", payload)
        
        if not data or "tasks" not in data:
            return {}

        for task in data.get("tasks", []):
            if task.get("result"):
                item = task["result"][0]
                return {
                    "volume": item.get("search_volume"),
                    "cpc": item.get("cpc"),
                    "competition": item.get("competition")
                }
        return {}

# Singleton instance
dfs_client = DataForSEOClient()
