import enum
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    CLIENT = "client"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.CLIENT)

    # For CLIENT users: links to the specific account they can view
    client_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
