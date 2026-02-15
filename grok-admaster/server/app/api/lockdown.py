
from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from datetime import datetime
import random

router = APIRouter()

# --- Models ---

class SystemIntegrity(BaseModel):
    core_systems: str # "CRITICAL" | "STABLE" | "WARNING"
    core_systems_val: int
    firewall_load: int
    encryption_layer: str # "STABLE" | "DEGRADED"
    network_integrity: int

class Countermeasure(BaseModel):
    id: str
    name: str
    info: str
    status: str # "Active" | "Inactive" | "Deploying"
    last_run: str
    icon: str

class SecurityLog(BaseModel):
    timestamp: str
    level: str # "INFO" | "WARNING" | "CRITICAL" | "ERROR"
    event: str
    details: str
    color: str

class LockdownStatus(BaseModel):
    active: bool
    integrity: SystemIntegrity
    countermeasures: List[Countermeasure]
    logs: List[SecurityLog]
    defcon: int

# --- Mock Data ---

MOCK_COUNTERMEASURES = [
    Countermeasure(id="c1", name="FLUSH SESSION CACHE", info="LAST RUN: 2 MIN AGO", status="Inactive", last_run="2 min ago", icon="cached"),
    Countermeasure(id="c2", name="CYCLE ADMIN KEYS", info="REQUIRES AUTH", status="Inactive", last_run="Never", icon="vpn_key"),
    Countermeasure(id="c3", name="ISOLATE SQL CLUSTERS", info="STATUS: VULNERABLE", status="Inactive", last_run="1 hour ago", icon="dns"),
    Countermeasure(id="c4", name="SEVER UPLINKS", info="EMERGENCY ONLY", status="Inactive", last_run="Never", icon="router"),
]

MOCK_LOGS = [
    SecurityLog(timestamp="23:42:01", level="WARNING", event="Auth_Fail", details="IP: 192.168.0.44", color="text-primary"),
    SecurityLog(timestamp="23:42:02", level="WARNING", event="Auth_Fail", details="IP: 192.168.0.44", color="text-primary"),
    SecurityLog(timestamp="23:42:05", level="CRITICAL", event="Port_Breach", details="Firewall breached port 443", color="text-primary"),
    SecurityLog(timestamp="23:42:06", level="WARNING", event="CPU_Spike", details="CPU Load 99%", color="text-yellow-500"),
]

# State (In-memory for demo)
GLOBAL_LOCKDOWN_STATE = {
    "active": False,
    "defcon": 5
}

# --- Endpoints ---

@router.get("/status", response_model=LockdownStatus)
async def get_lockdown_status():
    return LockdownStatus(
        active=GLOBAL_LOCKDOWN_STATE["active"],
        defcon=GLOBAL_LOCKDOWN_STATE["defcon"],
        integrity=SystemIntegrity(
            core_systems="CRITICAL" if GLOBAL_LOCKDOWN_STATE["active"] else "STABLE",
            core_systems_val=12 if GLOBAL_LOCKDOWN_STATE["active"] else 98,
            firewall_load=98,
            encryption_layer="STABLE",
            network_integrity=12 if GLOBAL_LOCKDOWN_STATE["active"] else 100
        ),
        countermeasures=MOCK_COUNTERMEASURES,
        logs=MOCK_LOGS
    )

@router.post("/toggle")
async def toggle_lockdown(active: bool):
    GLOBAL_LOCKDOWN_STATE["active"] = active
    GLOBAL_LOCKDOWN_STATE["defcon"] = 1 if active else 5
    
    # Add a log entry
    timestamp = datetime.now().strftime("%H:%M:%S")
    MOCK_LOGS.insert(0, SecurityLog(
        timestamp=timestamp,
        level="CRITICAL" if active else "INFO",
        event="LOCKDOWN_CHANGE",
        details=f"Lockdown {'INITIATED' if active else 'DEACTIVATED'}",
        color="text-red-500" if active else "text-green-500"
    ))
    
    return {"status": "success", "active": active}

@router.post("/countermeasure/{id}")
async def trigger_countermeasure(id: str):
    # Find and update mock
    for cm in MOCK_COUNTERMEASURES:
        if cm.id == id:
            cm.last_run = "Just now"
            # Add log
            MOCK_LOGS.insert(0, SecurityLog(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                level="INFO",
                event="COUNTERMEASURE",
                details=f"Executed {cm.name}",
                color="text-blue-500"
            ))
            return {"status": "executed", "countermeasure": cm}
    raise HTTPException(status_code=404, detail="Countermeasure not found")
