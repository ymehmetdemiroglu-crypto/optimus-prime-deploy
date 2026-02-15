import httpx
import json
import logging
from typing import Dict, Any, List, Optional
from app.core.config import settings

class OpenRouterClient:
    """Deterministic AI strategy client via OpenRouter."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.logger = logging.getLogger("openrouter_service")

    async def generate_strategy(self, context: Dict[str, Any], objective: str = "maximize_roas") -> Dict[str, Any]:
        """Generate advertising strategy based on market context."""
        
        prompt = f"""
        Analyze the following Amazon market data and provide a deterministic advertising strategy.
        Objective: {objective}
        
        Market Data:
        {json.dumps(context, indent=2)}
        
        Return ONLY a JSON object with this structure:
        {{
            "strategy_name": "string",
            "suggested_daily_budget": float,
            "recommended_keywords": [{{ "keyword": "string", "match_type": "EXACT|PHRASE|BROAD", "suggested_bid": float }}],
            "reasoning": "short explanation"
        }}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://grok-admaster.com", # Required by OpenRouter
            "X-Title": "Grok AdMaster",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert Amazon PPC Strategist. You output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": { "type": "json_object" }
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                result = response.json()
                
                content = result['choices'][0]['message']['content']
                return json.loads(content)
            except Exception as e:
                self.logger.error(f"OpenRouter strategy generation failed: {e}")
                # Return a safe fallback strategy
                return {
                    "strategy_name": "fallback_conservative",
                    "suggested_daily_budget": 20.0,
                    "recommended_keywords": [],
                    "reasoning": "Fallback due to AI service error"
                }
