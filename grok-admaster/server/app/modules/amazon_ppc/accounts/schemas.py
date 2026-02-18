from pydantic import BaseModel
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

class AccountBase(BaseModel):
    name: str
    amazon_account_id: str
    region: str = "NA"

class AccountCreate(AccountBase):
    pass

class AccountRead(AccountBase):
    id: int
    status: str
    created_at: datetime
    profiles: List[ProfileRead] = []

    class Config:
        from_attributes = True
