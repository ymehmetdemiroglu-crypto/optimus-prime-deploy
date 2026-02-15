from pydantic import BaseModel, EmailStr
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
    company_name: str
    primary_contact_email: Optional[EmailStr] = None

class AccountCreate(AccountBase):
    pass

class AccountRead(AccountBase):
    id: int
    created_at: datetime
    is_active: bool
    profiles: List[ProfileRead] = []

    class Config:
        from_attributes = True
