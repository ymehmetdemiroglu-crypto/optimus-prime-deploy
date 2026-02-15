"""
OpenRouter Multi-Model Client
-----------------------------
Provides access to multiple AI models via OpenRouter based on task role.
Supported Roles: Strategist (GPT-4), Creative (Claude Sonnet), Analyst (Claude Haiku), Fast (Llama 3).
"""
import os
import httpx
import logging
from typing import Optional, Dict, Any, List, Union
from enum import Enum

logger = logging.getLogger(__name__)

from app.core.config import settings

OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class ModelRole(str, Enum):
    """Functional roles that map to specific models."""
    STRATEGIST = "strategist"   # Deep reasoning, complex strategy (GPT-4)
    CREATIVE = "creative"       # Ad copy, marketing text (Claude 3.5 Sonnet)
    ANALYST = "analyst"         # Data parsing, log analysis (Claude 3 Haiku)
    FAST_CHAT = "fast_chat"     # Quick Q&A, simple tasks (Llama 3)


# Default Model Mapping
MODEL_MAP = {
    ModelRole.STRATEGIST: "openai/gpt-4-turbo",
    ModelRole.CREATIVE: "anthropic/claude-3.5-sonnet",
    ModelRole.ANALYST: "anthropic/claude-3-haiku",
    ModelRole.FAST_CHAT: "meta-llama/llama-3-8b-instruct:free"
}


async def call_ai_model(
    prompt: str,
    role: Union[ModelRole, str] = ModelRole.STRATEGIST,
    system_prompt: str = "You are an expert Amazon PPC AI assistant.",
    max_tokens: int = 1000,
    temperature: Optional[float] = None,
    specific_model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Call an AI model via OpenRouter. Automatically selects the best model for the role.
    
    Args:
        prompt: The user prompt
        role: Functional role (STRATEGIST, CREATIVE, etc.) to determine model
        system_prompt: System context
        max_tokens: Max output tokens
        temperature: Creativity (0.0=focused, 1.0=creative). dynamic default based on role.
        specific_model: Override to use a specific model ID string
    """
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set in environment")
        return {"error": "OpenRouter API key not configured", "content": None}
        
    # Determine Model
    model_id = specific_model or MODEL_MAP.get(role, MODEL_MAP[ModelRole.STRATEGIST])
    
    # Determine Default Temperature if not provided
    if temperature is None:
        if role == ModelRole.CREATIVE:
            temperature = 0.8
        elif role == ModelRole.ANALYST:
            temperature = 0.2
        else:
            temperature = 0.7  # Default balanced

    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://grok-admaster.app",
                    "X-Title": "Grok AdMaster PPC"
                },
                json=payload,
                timeout=60.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get("choices"):
                 return {"error": "No response choices returned", "content": None}

            return {
                "content": data["choices"][0]["message"]["content"],
                "model": data.get("model", model_id),
                "role_used": role,
                "usage": data.get("usage", {}),
                "error": None
            }
            
    except Exception as e:
        logger.error(f"AI Model Call Failed ({model_id}): {str(e)}")
        return {"error": str(e), "content": None}


# Backward compatibility alias
call_gpt4 = call_ai_model


async def call_gpt4_streaming(
    prompt: str,
    system_prompt: str = "You are an expert Amazon PPC strategist.",
    model: str = "openai/gpt-4-turbo"
):
    """
    Stream GPT-4 responses for real-time display.
    Yields chunks of text as they arrive.
    """
    if not OPENROUTER_API_KEY:
        yield {"error": "OpenRouter API key not configured"}
        return
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": True
            },
            timeout=60.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        import json
                        chunk = json.loads(data)
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
