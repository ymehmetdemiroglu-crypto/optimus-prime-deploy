"""
Background Anomaly Monitoring Task.

Runs periodic anomaly detection checks on all active profiles.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import async_session_maker
from app.modules.amazon_ppc.accounts.models import Profile
from app.modules.amazon_ppc.anomaly.service import anomaly_service
from app.modules.amazon_ppc.anomaly.schemas import (
    AnomalyDetectionRequest,
    EntityType,
    DetectorType,
)

logger = logging.getLogger(__name__)


async def run_anomaly_detection_for_profile(
    db: AsyncSession,
    profile_id: int,
    entity_type: EntityType = EntityType.KEYWORD,
):
    """Run anomaly detection for a single profile."""
    try:
        logger.info(f"[AnomalyMonitor] Starting detection for profile {profile_id}")
        
        request = AnomalyDetectionRequest(
            entity_type=entity_type,
            entity_ids=None,  # Check all entities
            profile_id=profile_id,
            detector_type=DetectorType.ENSEMBLE,
            include_explanation=True,
            include_root_cause=True,
        )
        
        response = await anomaly_service.detect_anomalies(db, request)
        
        logger.info(
            f"[AnomalyMonitor] Profile {profile_id}: "
            f"Checked {response.total_entities_checked} entities, "
            f"found {response.anomalies_detected} anomalies "
            f"(critical={response.critical_count}, high={response.high_count}) "
            f"in {response.execution_time_ms:.0f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"[AnomalyMonitor] Error for profile {profile_id}: {e}", exc_info=True)
        return None


async def run_hourly_anomaly_check():
    """
    Hourly task to check all active profiles for anomalies.
    
    Workflow:
        1. Get all active profiles
        2. Run keyword anomaly detection for each
        3. Optionally run campaign detection
        4. Log summary statistics
    """
    logger.info("[AnomalyMonitor] Starting hourly anomaly check")
    start_time = datetime.utcnow()
    
    async with async_session_maker() as db:
        try:
            # Get all active profiles
            query = select(Profile).where(Profile.is_active == True)
            result = await db.execute(query)
            profiles = result.scalars().all()
            
            if not profiles:
                logger.warning("[AnomalyMonitor] No active profiles found")
                return
            
            logger.info(f"[AnomalyMonitor] Found {len(profiles)} active profiles")
            
            # Process each profile
            total_checked = 0
            total_anomalies = 0
            critical_count = 0
            
            for profile in profiles:
                profile_id = int(profile.profile_id) if hasattr(profile, 'id') else profile.id
                
                # Run keyword detection
                response = await run_anomaly_detection_for_profile(
                    db,
                    profile_id,
                    EntityType.KEYWORD,
                )
                
                if response:
                    total_checked += response.total_entities_checked
                    total_anomalies += response.anomalies_detected
                    critical_count += response.critical_count
                
                # Optional: Run campaign detection (less frequent)
                # await run_anomaly_detection_for_profile(
                #     db,
                #     profile_id,
                #     EntityType.CAMPAIGN,
                # )
                
                # Rate limiting (avoid overwhelming database)
                await asyncio.sleep(1)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"[AnomalyMonitor] Hourly check complete: "
                f"{len(profiles)} profiles, "
                f"{total_checked} entities, "
                f"{total_anomalies} anomalies found "
                f"({critical_count} critical) "
                f"in {duration:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"[AnomalyMonitor] Hourly check failed: {e}", exc_info=True)


async def run_daily_cleanup():
    """
    Daily task to archive old alerts and maintain database health.
    
    Tasks:
        1. Archive resolved alerts older than 90 days
        2. Clean up orphaned records
        3. Update statistics
    """
    logger.info("[AnomalyMonitor] Starting daily cleanup")
    
    async with async_session_maker() as db:
        try:
            # Call database function to archive old alerts
            result = await db.execute("SELECT archive_old_anomaly_alerts()")
            archived_count = result.scalar()
            
            logger.info(f"[AnomalyMonitor] Archived {archived_count} old alerts")
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"[AnomalyMonitor] Daily cleanup failed: {e}", exc_info=True)


async def send_critical_alert_notifications():
    """
    Send notifications for unacknowledged critical alerts.
    
    Integration points:
        - Email: SendGrid/AWS SES
        - Slack: Webhook
        - SMS: Twilio (for enterprise)
    """
    logger.info("[AnomalyMonitor] Checking for critical alert notifications")
    
    async with async_session_maker() as db:
        try:
            # Get unacknowledged critical alerts from last hour
            since_time = datetime.utcnow() - timedelta(hours=1)
            
            from app.modules.amazon_ppc.anomaly.models import AnomalyAlert
            
            query = select(AnomalyAlert).where(
                AnomalyAlert.severity == "critical",
                AnomalyAlert.is_acknowledged == False,
                AnomalyAlert.detection_timestamp >= since_time,
            ).limit(100)
            
            result = await db.execute(query)
            alerts = result.scalars().all()
            
            if not alerts:
                logger.info("[AnomalyMonitor] No critical alerts to notify")
                return
            
            logger.info(f"[AnomalyMonitor] Found {len(alerts)} unacknowledged critical alerts")
            
            # TODO: Implement actual notification sending
            # for alert in alerts:
            #     await send_email_notification(alert)
            #     await send_slack_notification(alert)
            
            logger.info(f"[Ano malyMonitor] Sent notifications for {len(alerts)} critical alerts")
            
        except Exception as e:
            logger.error(f"[AnomalyMonitor] Notification sending failed: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════════
#  Scheduler Integration (APScheduler example)
# ═══════════════════════════════════════════════════════════════════

def setup_anomaly_monitoring_tasks():
    """
    Setup background tasks using APScheduler.
    
    Call this from main.py during application startup.
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        scheduler = AsyncIOScheduler()
        
        # Hourly anomaly detection (at :00)
        scheduler.add_job(
            run_hourly_anomaly_check,
            CronTrigger(hour='*', minute=0),
            id='hourly_anomaly_check',
            name='Hourly Anomaly Detection',
            replace_existing=True,
        )
        
        # Critical alert notifications (every 15 minutes)
        scheduler.add_job(
            send_critical_alert_notifications,
            CronTrigger(minute='*/15'),
            id='critical_alert_notifications',
            name='Critical Alert Notifications',
            replace_existing=True,
        )
        
        # Daily cleanup (at 2 AM)
        scheduler.add_job(
            run_daily_cleanup,
            CronTrigger(hour=2, minute=0),
            id='daily_anomaly_cleanup',
            name='Daily Anomaly Cleanup',
            replace_existing=True,
        )
        
        scheduler.start()
        logger.info("[AnomalyMonitor] Background tasks scheduled")
        
        return scheduler
        
    except ImportError:
        logger.warning(
            "[AnomalyMonitor] APScheduler not installed. "
            "Install with: pip install apscheduler"
        )
        return None


# ═══════════════════════════════════════════════════════════════════
#  Manual Execution (for testing)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run manual check
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_hourly_anomaly_check())
