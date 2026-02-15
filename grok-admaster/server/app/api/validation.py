
from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import asyncio
import random

router = APIRouter()

# --- Models ---

class ValidationLog(BaseModel):
    timestamp: str
    message: str
    status: str # "INFO" | "SUCCESS" | "WARNING" | "ERROR" | "input"
    details: Optional[str] = None

class ValidationStep(BaseModel):
    id: str
    name: str
    status: str # "pending" | "running" | "success" | "error"
    logs: List[ValidationLog]

class ValidationStatus(BaseModel):
    is_running: bool
    current_step: Optional[str] = None
    steps: List[ValidationStep]
    connection_strength: str # "EXCELLENT" | "GOOD" | "WEAK" | "OFFLINE"
    latency_ms: int
    encryption: str
    region: str

# --- Mock Data ---

MOCK_LOGS = [
    ValidationLog(timestamp="10:42:01", message="Initializing Handshake Protocol v4.2...", status="INFO"),
    ValidationLog(timestamp="10:42:01", message="Resolving host: sellingpartnerapi-na.amazon.com", status="INFO"),
    ValidationLog(timestamp="10:42:02", message="Verifying LWA Credentials...", status="SUCCESS", details="[OK]"),
    ValidationLog(timestamp="10:42:02", message="Checking IAM Role Policy Permissions...", status="INFO"),
    ValidationLog(timestamp="10:42:03", message="Testing SP-API Endpoint...", status="INFO"),
    ValidationLog(timestamp="10:42:05", message="SYSTEM SYNCHRONIZED", status="SUCCESS"),
]

# State
VALIDATION_STATE = {
    "is_running": False,
    "logs": MOCK_LOGS
}

# --- Endpoints ---

@router.get("/status", response_model=ValidationStatus)
async def get_validation_status():
    return ValidationStatus(
        is_running=VALIDATION_STATE["is_running"],
        current_step="diagnostics" if VALIDATION_STATE["is_running"] else None,
        steps=[],
        connection_strength="EXCELLENT",
        latency_ms=24,
        encryption="AES-256",
        region="US-EAST-1"
    )

@router.get("/logs", response_model=List[ValidationLog])
async def get_validation_logs():
    return VALIDATION_STATE["logs"]

@router.post("/run")
async def run_validation():
    # Reset logs or start new run logic
    VALIDATION_STATE["is_running"] = True
    # In a real app, this would trigger a background task
    return {"status": "started"}

@router.post("/stop")
async def stop_validation():
    VALIDATION_STATE["is_running"] = False
    return {"status": "stopped"}
