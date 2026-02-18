from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "client"
    client_account_id: Optional[int] = None


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    client_account_id: Optional[int] = None


class UserRead(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    client_account_id: Optional[int] = None
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True
