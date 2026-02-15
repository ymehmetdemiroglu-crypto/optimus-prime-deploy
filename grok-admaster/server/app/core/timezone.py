from datetime import datetime, timezone, timedelta
import pytz
from typing import Optional

class TimezoneManager:
    """Centralized timezone handling for production deployment."""
    
    def __init__(self, business_timezone: str = "America/New_York"):
        try:
            self.business_tz = pytz.timezone(business_timezone)
        except Exception:
            self.business_tz = pytz.UTC
        self.utc_tz = pytz.UTC
        
    def now_utc(self) -> datetime:
        """Get current time in UTC with timezone info."""
        return datetime.now(self.utc_tz)
        
    def now_business(self) -> datetime:
        """Get current time in business timezone."""
        return datetime.now(self.business_tz)
        
    def to_utc(self, dt: datetime) -> datetime:
        """Convert business time to UTC."""
        if dt.tzinfo is None:
            dt = self.business_tz.localize(dt)
        return dt.astimezone(self.utc_tz)
        
    def to_business(self, dt: datetime) -> datetime:
        """Convert UTC to business time."""
        if dt.tzinfo is None:
            dt = self.utc_tz.localize(dt)
        return dt.astimezone(self.business_tz)
        
    def schedule_at_business_time(self, hour: int, minute: int = 0) -> datetime:
        """Schedule task at business time (returns UTC)."""
        now_biz = self.now_business()
        target_biz = now_biz.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If already passed today, schedule for tomorrow
        if target_biz <= now_biz:
            target_biz += timedelta(days=1)
            
        return self.to_utc(target_biz)

# Global instance
tz_manager = TimezoneManager()
