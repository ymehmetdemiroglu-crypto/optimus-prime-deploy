"""
Chat API endpoints with AI simulation.
"""
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_simulator import generate_optimus_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message to Optimus AI and receive a simulated response."""
    try:
        response_content = generate_optimus_response(request.message, request.context_asin)
    except Exception as e:
        logger.error(f"AI simulator failed for message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate response")

    return ChatResponse(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        sender="optimus",
        content=response_content,
        timestamp=datetime.now(),
    )
