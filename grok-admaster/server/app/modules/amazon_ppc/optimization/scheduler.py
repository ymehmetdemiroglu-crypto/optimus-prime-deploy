"""
Scheduler for automated optimization runs.
Handles periodic optimization and rule-based triggers.
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

from ..models.ppc_data import PPCCampaign
from ..accounts.models import Account
from .engine import OptimizationEngine, OptimizationStrategy

logger = logging.getLogger(__name__)


class ScheduleFrequency(str, Enum):
    """How often to run optimization."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"


@dataclass
class OptimizationSchedule:
    """Configuration for scheduled optimization."""
    account_id: int
    campaign_ids: List[int]  # Empty = all campaigns
    strategy: OptimizationStrategy
    frequency: ScheduleFrequency
    target_acos: float = 25.0
    target_roas: float = 4.0
    auto_execute: bool = False  # If True, execute without approval
    min_confidence: float = 0.7
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


@dataclass 
class SchedulerState:
    """Runtime state of the scheduler."""
    is_running: bool = False
    schedules: List[OptimizationSchedule] = field(default_factory=list)
    run_history: List[Dict[str, Any]] = field(default_factory=list)


class OptimizationScheduler:
    """
    Manages scheduled optimization runs.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.state = SchedulerState()
        self._task: Optional[asyncio.Task] = None
    
    async def _get_session(self) -> AsyncSession:
        """Create a new database session."""
        engine = create_async_engine(self.database_url)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        return async_session()
    
    def add_schedule(self, schedule: OptimizationSchedule):
        """Add a new optimization schedule."""
        schedule.next_run = self._calculate_next_run(schedule.frequency)
        self.state.schedules.append(schedule)
        logger.info(f"Added schedule for account {schedule.account_id}, next run: {schedule.next_run}")
    
    def remove_schedule(self, account_id: int):
        """Remove schedules for an account."""
        self.state.schedules = [s for s in self.state.schedules if s.account_id != account_id]
    
    def _calculate_next_run(self, frequency: ScheduleFrequency) -> datetime:
        """Calculate next run time based on frequency."""
        now = datetime.now()
        
        if frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            # Next day at 6 AM
            next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        elif frequency == ScheduleFrequency.WEEKLY:
            # Next Monday at 6 AM
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
            return next_run + timedelta(days=days_until_monday)
        else:
            return now + timedelta(hours=4)
    
    async def run_scheduled_optimizations(self):
        """Check and run due optimizations."""
        now = datetime.now()
        
        for schedule in self.state.schedules:
            if not schedule.is_active:
                continue
            
            if schedule.next_run and schedule.next_run <= now:
                try:
                    result = await self._run_optimization(schedule)
                    schedule.last_run = now
                    schedule.next_run = self._calculate_next_run(schedule.frequency)
                    
                    self.state.run_history.append({
                        'account_id': schedule.account_id,
                        'run_time': now.isoformat(),
                        'result': result,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    logger.error(f"Scheduled optimization failed: {e}")
                    self.state.run_history.append({
                        'account_id': schedule.account_id,
                        'run_time': now.isoformat(),
                        'error': str(e),
                        'status': 'failed'
                    })
    
    async def _run_optimization(self, schedule: OptimizationSchedule) -> Dict[str, Any]:
        """Run optimization for a schedule."""
        async with await self._get_session() as db:
            engine = OptimizationEngine(db)
            
            # Get campaigns to optimize
            if schedule.campaign_ids:
                campaign_ids = schedule.campaign_ids
            else:
                # Get all active campaigns for account
                query = (
                    select(PPCCampaign.id)
                    .where(PPCCampaign.state == 'enabled')
                )
                result = await db.execute(query)
                campaign_ids = [row[0] for row in result.all()]
            
            all_results = []
            
            for campaign_id in campaign_ids:
                # Generate plan
                plan = await engine.generate_optimization_plan(
                    campaign_id=campaign_id,
                    strategy=schedule.strategy,
                    target_acos=schedule.target_acos,
                    target_roas=schedule.target_roas
                )
                
                # Execute if auto-execute enabled
                if schedule.auto_execute:
                    # Auto-approve all actions
                    for action in plan.actions:
                        action.approved = True
                    
                    result = await engine.execute_plan(
                        plan,
                        dry_run=False,
                        min_confidence=schedule.min_confidence
                    )
                else:
                    result = await engine.execute_plan(plan, dry_run=True)
                
                all_results.append({
                    'campaign_id': campaign_id,
                    'actions': plan.summary['total_actions'],
                    'executed': result['summary']['executed']
                })
            
            return {
                'campaigns_optimized': len(campaign_ids),
                'results': all_results
            }
    
    async def start(self, interval_seconds: int = 300):
        """Start the scheduler loop."""
        if self.state.is_running:
            logger.warning("Scheduler already running")
            return
        
        self.state.is_running = True
        logger.info(f"Starting optimization scheduler (interval: {interval_seconds}s)")
        
        async def loop():
            while self.state.is_running:
                try:
                    await self.run_scheduled_optimizations()
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(interval_seconds)
        
        self._task = asyncio.create_task(loop())
    
    async def stop(self):
        """Stop the scheduler."""
        self.state.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Optimization scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'is_running': self.state.is_running,
            'active_schedules': len([s for s in self.state.schedules if s.is_active]),
            'total_schedules': len(self.state.schedules),
            'recent_runs': self.state.run_history[-10:],
            'schedules': [
                {
                    'account_id': s.account_id,
                    'strategy': s.strategy.value,
                    'frequency': s.frequency.value,
                    'auto_execute': s.auto_execute,
                    'is_active': s.is_active,
                    'last_run': s.last_run.isoformat() if s.last_run else None,
                    'next_run': s.next_run.isoformat() if s.next_run else None
                }
                for s in self.state.schedules
            ]
        }
