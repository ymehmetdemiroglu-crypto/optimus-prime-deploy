"""
Ingestion Manager - orchestrates data collection across all accounts and profiles.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional
from datetime import datetime, timedelta, date
import logging
import asyncio

from ..accounts.models import Account, Credential, Profile
from .client import AmazonAdsAPIClient
from .etl import AmazonAdsETL

logger = logging.getLogger(__name__)

class IngestionManager:
    """
    Orchestrates ingestion across all active accounts and profiles.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.etl = AmazonAdsETL(db)
    
    async def sync_all_accounts(self):
        """
        Main entry point: sync data for all active accounts.
        """
        logger.info("Starting ingestion for all accounts")
        
        # Get all active accounts with credentials
        query = (
            select(Account)
            .where(Account.is_active == True)
            .options(selectinload(Account.credentials), selectinload(Account.profiles))
        )
        result = await self.db.execute(query)
        accounts = result.scalars().all()
        
        logger.info(f"Found {len(accounts)} active accounts")
        
        for account in accounts:
            try:
                await self.sync_account(account)
            except Exception as e:
                logger.error(f"Failed to sync account {account.id}: {e}")
                continue
    
    async def sync_account(self, account: Account):
        """
        Sync data for a single account.
        """
        logger.info(f"Syncing account: {account.company_name} (ID: {account.id})")
        
        # Get credentials
        if not account.credentials or len(account.credentials) == 0:
            logger.warning(f"No credentials found for account {account.id}")
            return
        
        cred = account.credentials[0]  # Use the first credential
        
        # Initialize API client
        client = AmazonAdsAPIClient(
            client_id=cred.client_id,
            client_secret=cred.client_secret,
            refresh_token=cred.refresh_token
        )
        
        try:
            # First, sync profiles if we don't have any
            if not account.profiles or len(account.profiles) == 0:
                await self.sync_profiles(account, client)
                await self.db.refresh(account, attribute_names=["profiles"])
            
            # Sync campaigns and keywords for each profile
            for profile in account.profiles:
                if profile.is_active:
                    await self.sync_profile_data(profile, client)
        finally:
            await client.close()
    
    async def sync_profiles(self, account: Account, client: AmazonAdsAPIClient):
        """
        Fetch and store profiles from the API.
        """
        logger.info(f"Fetching profiles for account {account.id}")
        
        try:
            profiles_data = await client.get_profiles()
            
            for profile_raw in profiles_data:
                profile_id = str(profile_raw.get("profileId"))
                
                # Check if profile exists
                query = select(Profile).where(Profile.profile_id == profile_id)
                result = await self.db.execute(query)
                existing = result.scalars().first()
                
                if not existing:
                    profile = Profile(
                        profile_id=profile_id,
                        account_id=account.id,
                        country_code=profile_raw.get("countryCode", "US"),
                        currency_code=profile_raw.get("currencyCode", "USD"),
                        timezone=profile_raw.get("timezone", "America/Los_Angeles"),
                        account_info_id=str(profile_raw.get("accountInfo", {}).get("id", ""))
                    )
                    self.db.add(profile)
            
            await self.db.commit()
            logger.info(f"Synced {len(profiles_data)} profiles")
        except Exception as e:
            logger.error(f"Failed to sync profiles: {e}")
            raise
    
    async def sync_profile_data(self, profile: Profile, client: AmazonAdsAPIClient):
        """
        Sync campaigns, keywords, and performance data for a single profile.
        """
        logger.info(f"Syncing data for profile {profile.profile_id}")
        
        try:
            # Fetch campaigns
            campaigns = await client.get_campaigns(profile.profile_id)
            await self.etl.load_campaigns(profile.profile_id, campaigns)
            
            # Fetch keywords
            keywords = await client.get_keywords(profile.profile_id)
            await self.etl.load_keywords(keywords)
            
            # Optionally, request performance reports
            # (Reports are async, so we'd need to poll for completion)
            await self.request_performance_reports(profile, client)
            
        except Exception as e:
            logger.error(f"Failed to sync profile {profile.profile_id}: {e}")
            raise
    
    async def request_performance_reports(
        self, 
        profile: Profile, 
        client: AmazonAdsAPIClient,
        report_date: Optional[date] = None
    ):
        """
        Request performance reports for yesterday.
        Note: This is async - reports take time to generate.
        You'd typically store report IDs and poll for completion in a background job.
        """
        if report_date is None:
            report_date = date.today() - timedelta(days=1)
        
        report_date_str = report_date.strftime("%Y%m%d")
        
        metrics = [
            "campaignId",
            "impressions",
            "clicks",
            "cost",
            "attributedSales14d",
            "attributedConversions14d"
        ]
        
        try:
            report_id = await client.create_report_request(
                profile_id=profile.profile_id,
                report_type="campaigns",
                report_date=report_date_str,
                metrics=metrics
            )
            
            logger.info(f"Created report request {report_id} for profile {profile.profile_id}")
            
            # In production, you'd store this report_id and poll for completion
            # For now, we'll just log it
            
        except Exception as e:
            logger.error(f"Failed to request report: {e}")
    
    async def poll_and_process_report(
        self, 
        profile: Profile, 
        client: AmazonAdsAPIClient, 
        report_id: str,
        report_date: date
    ):
        """
        Poll for report completion and process when ready.
        This would typically run as a background task.
        """
        max_attempts = 10
        wait_seconds = 30
        
        for attempt in range(max_attempts):
            try:
                report_status = await client.get_report(profile.profile_id, report_id)
                status = report_status.get("status")
                
                if status == "SUCCESS":
                    download_url = report_status.get("location")
                    if download_url:
                        metrics = await client.download_report(download_url)
                        await self.etl.load_campaign_performance(metrics, report_date)
                        logger.info(f"Successfully processed report {report_id}")
                        return
                elif status == "FAILURE":
                    logger.error(f"Report {report_id} failed")
                    return
                else:
                    # Still processing
                    await asyncio.sleep(wait_seconds)
            except Exception as e:
                logger.error(f"Error polling report {report_id}: {e}")
                return
        
        logger.warning(f"Report {report_id} did not complete in time")
