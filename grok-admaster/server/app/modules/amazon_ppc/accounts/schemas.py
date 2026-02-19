from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ProfileBase(BaseModel):
    profile_id: str
    country_code: str
    currency_code: str
    timezone: str
    account_info_id: str
    is_active: bool = True


class ProfileRead(ProfileBase):
    account_id: int

    class Config:
        from_attributes = True


class CredentialBase(BaseModel):
    client_id: str
    client_secret: str
    refresh_token: str


class CredentialCreate(CredentialBase):
    account_id: int


class AccountCreate(BaseModel):
    """Payload for creating a new client account."""
    name: str = Field(..., min_length=1, max_length=255, description="Account display name")
    # amazon_account_id is assigned after the first Amazon API profile sync
    amazon_account_id: Optional[str] = Field(
        default=None, description="Amazon seller account ID (populated after sync)"
    )
    region: str = Field(default="NA", description="Amazon marketplace region")


class AccountRead(BaseModel):
    """Full account representation returned by the API."""
    id: int
    name: str
    amazon_account_id: Optional[str] = None
    region: str
    status: str
    created_at: datetime
    profiles: List[ProfileRead] = []

    class Config:
        from_attributes = True
