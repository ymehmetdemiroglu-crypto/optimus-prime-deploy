from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    amazon_account_id = Column(String, unique=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    region = Column(String, nullable=False, default="NA")
    status = Column(String, default="onboarding")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    profiles = relationship("Profile", back_populates="account", cascade="all, delete-orphan")
    credentials = relationship("Credential", back_populates="account", cascade="all, delete-orphan")

class Profile(Base):
    """
    Represents a specific Amazon Ads Profile (e.g., US Market, UK Market).
    Corresponds to Amazon's 'profileId'.
    """
    __tablename__ = "profiles"

    profile_id = Column(String, primary_key=True, index=True) # Amazon numeric ID as string
    account_id = Column(Integer, ForeignKey("accounts.id"))
    country_code = Column(String) # US, UK, DE, etc.
    currency_code = Column(String) # USD, GBP, etc.
    timezone = Column(String)
    account_info_id = Column(String) # Amazon 'entityId' or similar
    
    # Status
    is_active = Column(Boolean, default=True)
    
    account = relationship("Account", back_populates="profiles")
    anomaly_alerts = relationship("AnomalyAlert", back_populates="profile", cascade="all, delete-orphan")
    anomaly_history = relationship("AnomalyHistory", back_populates="profile", cascade="all, delete-orphan")
    # Future relationship: campaigns = relationship("Campaign", back_populates="profile")

class Credential(Base):
    """
    Stores API tokens. Sensitive fields should be encrypted before storage.
    """
    __tablename__ = "credentials"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"))
    
    # Store these as encrypted strings in production
    client_id = Column(String, nullable=False)
    client_secret = Column(String, nullable=False)
    refresh_token = Column(String, nullable=False)
    
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    account = relationship("Account", back_populates="credentials")
