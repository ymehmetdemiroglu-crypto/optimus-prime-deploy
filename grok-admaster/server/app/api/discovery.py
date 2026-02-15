
from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# --- Models ---

class DiscoveryLog(BaseModel):
    timestamp: str
    message: str
    type: str = "info" # info, success, warning, error

class DiscoveredProfile(BaseModel):
    id: str
    profile_id: str
    region: str
    region_code: str
    country_code: str
    marketplace_id: str
    status: str # Ready, Conflict, Pending
    flag_url: str
    last_sync: str
    conflict_details: Optional[str] = None

class DiscoveryStats(BaseModel):
    total_found: int
    ready_to_sync: int
    conflicts: int
    pending: int

class SyncRequest(BaseModel):
    profile_ids: List[str]

# --- Mock Data ---

MOCK_LOGS = [
    DiscoveryLog(timestamp="10:42:01", message="Initializing global scan...", type="info"),
    DiscoveryLog(timestamp="10:42:01", message="Region: us-east-1 OK", type="info"),
    DiscoveryLog(timestamp="10:42:02", message="Found Marketplace: ATVPDKIKX0DER (US)", type="success"),
    DiscoveryLog(timestamp="10:42:02", message="Authenticating profile 459-2938-102...", type="info"),
    DiscoveryLog(timestamp="10:42:03", message="Profile verified. Status: Ready.", type="success"),
    DiscoveryLog(timestamp="10:42:05", message="Region: eu-west-2 OK", type="info"),
    DiscoveryLog(timestamp="10:42:06", message="Found Marketplace: A1F83G8C2ARO7P (UK)", type="success"),
    DiscoveryLog(timestamp="10:42:08", message="WARN: Profile ID conflict in eu-central-1", type="warning"),
    DiscoveryLog(timestamp="10:42:08", message="Error code: E_DUP_ENTRY_991", type="error"),
]

MOCK_PROFILES = [
    DiscoveredProfile(
        id="p1",
        profile_id="459-2938-102",
        region="North America",
        region_code="US-EAST-1",
        country_code="US",
        marketplace_id="ATVPDKIKX0DER",
        status="Ready",
        flag_url="https://lh3.googleusercontent.com/aida-public/AB6AXuBqlZLP5hP8tI7yUgGBBAXhb5DAaZ0Q-SyoAMyXjKTLeUEhbZEzH_6JuW1vLb3Fi01ZZyeNGiTc8Hj4Iu53HcDmRdw6mAGtRDfgIpnAntoENDOqSVw0EhmEj0L__213Eo9YeYtH2i3v2sVkBYbD3cAxsbVHRfyN5siXtAf2hB2-N-HK7OpkD4IoX5RjToB0oeJLkrU_ezICll7xa1XpVGSxuk3cTj7UVrbUsUim-cJXycaSxffu_ASFJl2lTDlS2eyYyGTlYULLCykS",
        last_sync="2h ago"
    ),
    DiscoveredProfile(
        id="p2",
        profile_id="882-1092-441",
        region="United Kingdom",
        region_code="EU-WEST-2",
        country_code="GB",
        marketplace_id="A1F83G8C2ARO7P",
        status="Ready",
        flag_url="https://lh3.googleusercontent.com/aida-public/AB6AXuAE0e6jInin49wVanEMJNTg3lwEqrQx_-AFnILHMZanLUgutroKXyCrTwT-5NWlucqKaSrtF9tfOT4prMZr64-uEvZQ-MMaX2BmU65PmarR_wsCCCOJQMEkb9jxnYxgkFnZqngb0Lx1NmNrVH19q5qau-PCPFrfleQMRn2BWNSnicg8JvmwsaFag03qYu2oOdVxdd0MYqVQJQrD-R-GF0TN06bH0QfgtdMQAbM-dodnC8waiswdgPMxAN5b-pj2Ms12oir__M2LU-ie",
        last_sync="Never"
    ),
    DiscoveredProfile(
        id="p3",
        profile_id="991-2331-001",
        region="Germany",
        region_code="EU-CENTRAL-1",
        country_code="DE",
        marketplace_id="A1PA6795UKMFR9",
        status="Conflict",
        flag_url="https://lh3.googleusercontent.com/aida-public/AB6AXuA7gzg3V7vYGqHRiXIj45-ES-iKlHqYmzP8xfRaC38P0GEaFE8nWIKQ0yMp59eluzwxe4-gAM063mHV-FzWsyuEto1fKIDCXpv96YdBAX4O5snm-1rIgkX3z6chwuANq887kRrdL9xQtilEYOEbI-7Wi7oLObThXN4VYzklJdSzKM9rVW91z458BWBRMgoYF6iqUbdP5x8Ao8UC1kupn5Ydq8pw3z9j_asAhqZ78ssR1IXsqCQuZuTqi_uJ09Q1mN3QM8Mc1UTXx_bK",
        last_sync="1d ago",
        conflict_details="Duplicate Profile ID detected in existing database."
    ),
    DiscoveredProfile(
        id="p4",
        profile_id="102-4458-992",
        region="Japan",
        region_code="AP-NORTHEAST-1",
        country_code="JP",
        marketplace_id="A1VC38T7YXB528",
        status="Ready",
        flag_url="https://lh3.googleusercontent.com/aida-public/AB6AXuB2SzGP937zOqr2Wyz7sWpEzfz4sSsmIItKnrqS8Wpf9SykAU6DC6b53YTp1HvXffV2YLLosryplbJDAVIIFE5ShySoGYlvcw4r_mrCSusvWXZJasOdGmUSh8zy8sp1VXvKgqieNTE6ncgUz0soSnYcStt4QipmWR0wbEYKWrtLYTTkcEIDzeqWLdbUlpKICM6pVrahO8nShx8a70FUqBsoBxxSDigbiVNSwfU3hubkEYMRfOYGc324youF5JAAy3YaaQdHV3wlxOUE",
        last_sync="1d ago"
    ),
    DiscoveredProfile(
        id="p5",
        profile_id="Scanning...",
        region="Brazil",
        region_code="SA-EAST-1",
        country_code="BR",
        marketplace_id="--",
        status="Pending",
        flag_url="https://lh3.googleusercontent.com/aida-public/AB6AXuAiRS0o9Tn7KdyXr_KjBgqWzaJatrdBLflql41mag2GM9CFdwi_QmyxeG45YZEQQ7mj_LfyfQtNnHxvSgMACN8pM2EYYJ1kHQjLLe1UjLo_k-_FpEIQOVrd98_mRsJ_nK-EbDsG8E4SY1VDMjf44dzF8Z1WtmA2zPhjgqI27icRNzDHfMVUSfC9ICzfdWToN6DMEF6xzO8QnSOkhEDOKtDMiDCHULZvftb4y0cgZJApaUT58qmtRYFiMMVdewnV3qeEzk1_horwyUxF",
        last_sync="Never"
    )
]

MOCK_STATS = DiscoveryStats(
    total_found=12,
    ready_to_sync=8,
    conflicts=3,
    pending=1
)

# --- Endpoints ---

@router.get("/profiles", response_model=List[DiscoveredProfile])
async def get_discovered_profiles():
    return MOCK_PROFILES

@router.get("/stats", response_model=DiscoveryStats)
async def get_discovery_stats():
    return MOCK_STATS

@router.get("/logs", response_model=List[DiscoveryLog])
async def get_discovery_logs():
    return MOCK_LOGS

@router.post("/sync")
async def sync_profiles(request: SyncRequest):
    return {"status": "success", "message": f"Sync initialized for {len(request.profile_ids)} profiles"}
