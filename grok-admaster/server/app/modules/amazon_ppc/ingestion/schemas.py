"""
Pydantic schemas for Amazon Ads API data structures.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal

# Campaign Schemas
class CampaignResponse(BaseModel):
    campaignId: str
    name: str
    campaignType: str
    targetingType: str
    state: str
    dailyBudget: float
    startDate: Optional[str] = None
    endDate: Optional[str] = None

class KeywordResponse(BaseModel):
    keywordId: str
    campaignId: str
    adGroupId: Optional[str] = None
    keywordText: str
    matchType: str
    state: str
    bid: float

# Reporting Schemas
class CampaignMetrics(BaseModel):
    campaignId: str
    date: str
    impressions: int = 0
    clicks: int = 0
    cost: float = 0.0
    attributedSales14d: float = 0.0
    attributedConversions14d: int = 0

class KeywordMetrics(BaseModel):
    campaignId: str
    keywordId: str
    date: str
    impressions: int = 0
    clicks: int = 0
    cost: float = 0.0
    attributedSales14d: float = 0.0
    attributedConversions14d: int = 0

# Profile Schema
class ProfileResponse(BaseModel):
    profileId: str
    countryCode: str
    currencyCode: str
    timezone: str
    accountInfo: dict
