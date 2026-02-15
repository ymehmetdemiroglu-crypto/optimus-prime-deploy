"""
Ad Copy Generator Service
-------------------------
Uses the 'CREATIVE' AI model (Claude 3.5 Sonnet) to generate high-converting
Amazon ad headlines and custom product descriptions.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.services.openrouter_client import call_ai_model, ModelRole

logger = logging.getLogger(__name__)


async def generate_ad_headlines(
    product_name: str,
    keywords: List[str],
    unique_selling_points: List[str],
    target_audience: str = "general",
    tone: str = "persuasive"
) -> Dict[str, Any]:
    """
    Generate Sponsored Brands headlines using the Creative AI model.
    """
    
    prompt = f"""Write 5 high-converting Amazon Sponsored Brands headlines for the following product.

Product: {product_name}
Target Keywords: {', '.join(keywords)}
Key Features: {', '.join(unique_selling_points)}
Target Audience: {target_audience}
Tone: {tone}

Constraints:
- Max 50 characters per headline
- Must be catchy and click-worthy
- No prohibited claims (e.g. "Best Seller", "#1")
- Focus on benefits, not just features

Format the output as a JSON list of strings.
"""

    result = await call_ai_model(
        prompt=prompt,
        role=ModelRole.CREATIVE,  # Uses Claude 3.5 Sonnet for better writing
        system_prompt="You are a world-class copywriter specializing in Amazon PPC advertising.",
        temperature=0.8,  # Higher creativity
        max_tokens=600
    )
    
    return {
        "headlines": result.get("content"),
        "model_used": result.get("model"),
        "role": result.get("role_used")
    }


async def improve_product_description(
    current_description: str,
    focus_keywords: List[str]
) -> Dict[str, Any]:
    """
    Rewrite a product description to be more engaging and SEO-optimized.
    """
    prompt = f"""Rewrite this Amazon product description to be more persuasive and include the focus keywords naturally.

Current Description:
{current_description}

Focus Keywords to include: {', '.join(focus_keywords)}

Requirements:
- Use bullet points for readability
- Focus on emotional benefits
- Keep the tone professional but exciting
- Highlight key value propositions
"""

    result = await call_ai_model(
        prompt=prompt,
        role=ModelRole.CREATIVE,
        max_tokens=1000
    )
    
    return {
        "improved_description": result.get("content"),
        "model_used": result.get("model")
    }
