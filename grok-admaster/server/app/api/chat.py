"""
Chat API endpoints with AI simulation.
"""
from fastapi import APIRouter
from datetime import datetime
import asyncio
import uuid
import random

from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_simulator import generate_optimus_response

router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message to Optimus AI and receive a simulated response."""
    # Simulate AI "thinking" time (1-2 seconds)
    await asyncio.sleep(random.uniform(1.0, 2.0))
    
    # Generate AI response
    response_content = generate_optimus_response(request.message, request.context_asin)
    
    return ChatResponse(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        sender="optimus",
        content=response_content,
        timestamp=datetime.now()
    )
