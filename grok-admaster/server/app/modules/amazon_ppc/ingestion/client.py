"""
Amazon Ads API Client.
Handles authentication, rate limiting, and API interactions.
"""
import aiohttp
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AmazonAdsAPIClient:
    """
    Async client for Amazon Advertising API.
    Supports multi-tenant operations with per-profile credentials.
    """
    
    BASE_URL = "https://advertising-api.amazon.com"
    TOKEN_URL = "https://api.amazon.com/auth/o2/token"
    
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure we have an active aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _refresh_access_token(self):
        """Refresh the access token using the refresh token."""
        await self._ensure_session()
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        async with self._session.post(self.TOKEN_URL, data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to refresh token: {resp.status} - {text}")
            
            result = await resp.json()
            self.access_token = result["access_token"]
            expires_in = result.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)  # 5 min buffer
            logger.info("Access token refreshed successfully")
    
    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        if not self.access_token or not self.token_expires_at or datetime.now() >= self.token_expires_at:
            await self._refresh_access_token()
        return self.access_token
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        profile_id: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated API request.
        Handles rate limiting with exponential backoff.
        """
        await self._ensure_session()
        token = await self._get_access_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Amazon-Advertising-API-ClientId": self.client_id,
            "Amazon-Advertising-API-Scope": profile_id,
            "Content-Type": "application/json"
        }
        
        url = f"{self.BASE_URL}{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self._session.request(
                    method, url, headers=headers, params=params, json=json_data
                ) as resp:
                    if resp.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry.")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if resp.status >= 400:
                        text = await resp.text()
                        logger.error(f"API Error {resp.status}: {text}")
                        raise Exception(f"API request failed: {resp.status} - {text}")
                    
                    return await resp.json()
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    # API Methods
    
    async def get_profiles(self) -> List[Dict[str, Any]]:
        """Fetch all profiles accessible with the current credentials."""
        return await self._make_request("GET", "/v2/profiles", profile_id="0")
    
    async def get_campaigns(self, profile_id: str, state_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch campaigns for a specific profile."""
        params = {"stateFilter": state_filter} if state_filter else {}
        return await self._make_request("GET", "/v2/sp/campaigns", profile_id, params=params)
    
    async def get_keywords(self, profile_id: str, campaign_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch keywords for a specific profile/campaign."""
        params = {}
        if campaign_id:
            params["campaignIdFilter"] = campaign_id
        return await self._make_request("GET", "/v2/sp/keywords", profile_id, params=params)
    
    async def create_report_request(
        self, 
        profile_id: str, 
        report_type: str,
        report_date: str,
        metrics: List[str]
    ) -> str:
        """
        Create a report request and return the report ID.
        report_type: e.g., 'campaigns', 'keywords'
        report_date: YYYYMMDD format
        """
        endpoint = f"/v2/sp/{report_type}/report"
        
        payload = {
            "reportDate": report_date,
            "metrics": ",".join(metrics)
        }
        
        response = await self._make_request("POST", endpoint, profile_id, json_data=payload)
        return response.get("reportId")
    
    async def get_report(self, profile_id: str, report_id: str) -> Dict[str, Any]:
        """Check the status of a report request."""
        endpoint = f"/v2/reports/{report_id}"
        return await self._make_request("GET", endpoint, profile_id)
    
    async def download_report(self, download_url: str) -> List[Dict[str, Any]]:
        """Download and parse a completed report."""
        await self._ensure_session()
        
        async with self._session.get(download_url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download report: {resp.status}")
            
            # Reports are typically gzip-compressed JSON
            import gzip
            import json
            content = await resp.read()
            decompressed = gzip.decompress(content)
            return json.loads(decompressed)
