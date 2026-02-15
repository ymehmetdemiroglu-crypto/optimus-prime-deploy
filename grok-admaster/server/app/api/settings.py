
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# --- Models ---

class AdminUser(BaseModel):
    id: str
    name: str
    email: str
    role: str = "Admin" # Admin, Super Admin
    avatar: str
    active: bool = True
    two_factor_enabled: bool = False
    last_active: str

class AdminUserCreate(BaseModel):
    name: str
    email: str
    role: str = "Admin"

class SystemSettings(BaseModel):
    # Throttling
    requests_per_second: int
    burst_capacity: int
    rate_limit_strategy: str # Token Bucket, Leaky Bucket, Fixed Window
    strict_rate_limiting: bool
    
    # Notifications
    slack_webhook: Optional[str] = None
    alert_email: Optional[str] = None
    notify_downtime: bool = True
    notify_api_spike: bool = True
    notify_new_admin: bool = True
    notify_daily_digest: bool = True

class BackupStatus(BaseModel):
    last_backup: str
    server_load: int

# --- Mock Data ---

MOCK_USERS = [
    AdminUser(
        id="u1",
        name="Cipher Admin",
        email="super.admin@optimus.sys",
        role="Super Admin",
        avatar="https://lh3.googleusercontent.com/aida-public/AB6AXuDwO6VCt9A-t3llEptYfCnZRlzL2fD8zmXzwRb1zEA3OLA2sVmY6Bdgo06Vmxl2lpAVaxWT9vqzp9CjdGixov7dtdcDurzNKMESs4_sZMzhpM5mEBSdCVq_a8SvnS-OwPPQBdVRNp-5sqKb6h4Xy74egGl8dUnpMOjqAx8kBRK8v4I_UvoYHLkVeQ12d6jbvb7F4Hgd99S_c8RFJnZ2rJhU88AQdRGuhCfHkD9jhBNWgSGWuROSV3jJ4oi4pKvnJXBlvBcKaM3Hjkpr",
        active=True,
        two_factor_enabled=True,
        last_active="2m ago"
    ),
    AdminUser(
        id="u2",
        name="Sarah Connor",
        email="sarah.c@optimus.sys",
        role="Admin",
        avatar="https://lh3.googleusercontent.com/aida-public/AB6AXuBxiyamCEvuNNHDbi0HeheaFe-4M0wlhidt8W4xvOdv9Nq4S2-X1vacZuV-QH-EiN9V1dLHz5zQYSGXyMbL3HNhwJi2cRvE9SeUoXHb0BOKOzE_a4d_hnWe14HDApN7XRgPTguoCvO0UvfnxIfpcsYQScDd41zL4ZEX_jwUFK2P2iosBMD4CkwSj0BbqqDYx86uIKjcUay8fiMc_I-kbCUNJi1EHfx_8FsEkQxD-DSM4valrR9UJNKCPr-TdzZlZyqUEJKQ1GHA0TdG",
        active=True,
        two_factor_enabled=False,
        last_active="4h ago"
    ),
    AdminUser(
        id="u3",
        name="John Doe",
        email="j.doe@optimus.sys",
        role="Admin",
        avatar="https://lh3.googleusercontent.com/aida-public/AB6AXuA_irq6eruPM6_W6iEqHC0vH36HuEqBrhaEpZxdAccsUHfzIKoJS8sJsSTVPT_wce5uLTvBdDTlxXeCwpBsLHdGgv8alm_gbaZuIla_BUT1GEs5nbohitsOb7OqowrGHPHBjXMx_dhhtydK-A9Z-S4TpsqXQ-8EvkDlAA6q1zURTpfTF3Uj1_POBRClcGHzIiOHAQ4VxYTrNyU1zvdsy2cpk20EK0pEMUmuPG1CRgKt6tiAxo5P264Fl2N30u_dBYD02IKPUnHUarW2",
        active=False,
        two_factor_enabled=False,
        last_active="14d ago"
    )
]

CURRENT_SETTINGS = SystemSettings(
    requests_per_second=4500,
    burst_capacity=500,
    rate_limit_strategy="Token Bucket",
    strict_rate_limiting=True,
    slack_webhook="",
    alert_email="ops-team@optimus.sys",
    notify_downtime=True,
    notify_api_spike=True,
    notify_new_admin=False,
    notify_daily_digest=True
)

# --- Endpoints ---

@router.get("/admins", response_model=List[AdminUser])
async def get_admins():
    return MOCK_USERS

@router.post("/admins", response_model=AdminUser)
async def create_admin(user: AdminUserCreate):
    new_user = AdminUser(
        id=f"u{len(MOCK_USERS) + 1}",
        name=user.name,
        email=user.email,
        role=user.role,
        avatar="", # Placeholder
        active=True,
        two_factor_enabled=False,
        last_active="Just now"
    )
    MOCK_USERS.append(new_user)
    return new_user

@router.delete("/admins/{user_id}")
async def revoke_admin(user_id: str):
    global MOCK_USERS
    MOCK_USERS = [u for u in MOCK_USERS if u.id != user_id]
    return {"status": "revoked"}

@router.get("/config", response_model=SystemSettings)
async def get_sytem_settings():
    return CURRENT_SETTINGS

@router.put("/config", response_model=SystemSettings)
async def update_system_settings(settings: SystemSettings):
    global CURRENT_SETTINGS
    CURRENT_SETTINGS = settings
    return CURRENT_SETTINGS

@router.get("/backup-status", response_model=BackupStatus)
async def get_backup_status():
    return BackupStatus(
        last_backup=datetime.now().strftime("%I:%M %p"),
        server_load=42
    )
