import argparse
import sys
import os
import json
import asyncio
from dotenv import load_dotenv

# Path setup
server_path = os.path.abspath(os.path.join(os.getcwd(), 'grok-admaster', 'server'))
sys.path.append(server_path)

# Load env from server directory
env_path = os.path.join(server_path, '.env')
load_dotenv(env_path)

try:
    from sqlalchemy.future import select
    from app.core.database import AsyncSessionLocal
    from app.models.campaign import Campaign
except ImportError as e:
    print(f"Error importing app modules: {e}")
    sys.exit(1)

async def list_campaigns():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Campaign))
        campaigns = result.scalars().all()
        return [{
            "id": str(c.id), 
            "name": c.name, 
            "status": c.state, 
            "ai_mode": c.ai_mode,
            "acos": float(c.current_acos) if c.current_acos else 0.0
        } for c in campaigns]

async def get_campaign(campaign_id):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Campaign).where(Campaign.id == campaign_id))
        campaign = result.scalar_one_or_none()
        if not campaign:
            return {"error": "Campaign not found"}
        return {
            "id": str(campaign.id),
            "name": campaign.name,
            "status": campaign.state,
            "ai_mode": campaign.ai_mode,
            "budget": float(campaign.daily_budget) if campaign.daily_budget else 0.0
        }

async def update_strategy(campaign_id, ai_mode):
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Campaign).where(Campaign.id == campaign_id))
        campaign = result.scalar_one_or_none()
        if not campaign:
            return {"error": "Campaign not found"}
        
        campaign.ai_mode = ai_mode
        await session.commit()
        await session.refresh(campaign)
        return {
            "id": str(campaign.id),
            "new_mode": campaign.ai_mode,
            "status": "success"
        }

def main():
    parser = argparse.ArgumentParser(description="Campaign Manager Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List
    subparsers.add_parser("list", help="List all campaigns")

    # Get
    get_parser = subparsers.add_parser("get", help="Get a campaign details")
    get_parser.add_argument("id", help="Campaign ID")

    # Update
    update_parser = subparsers.add_parser("update_strategy", help="Update AI strategy")
    update_parser.add_argument("id", help="Campaign ID")
    update_parser.add_argument("mode", help="New AI Mode (e.g., 'profit_guard', 'aggressive')")

    args = parser.parse_args()

    if args.command == "list":
        result = asyncio.run(list_campaigns())
    elif args.command == "get":
        result = asyncio.run(get_campaign(args.id))
    elif args.command == "update_strategy":
        result = asyncio.run(update_strategy(args.id, args.mode))
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
