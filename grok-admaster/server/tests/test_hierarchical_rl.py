
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import text
from app.modules.amazon_ppc.ml.hierarchical_rl import HierarchicalBudgetController, PortfolioState

@pytest.mark.asyncio
async def test_hierarchical_allocation_flow():
    # Mock DB Session
    mock_db = AsyncMock()
    
    # Mock execute result for _build_portfolio_state
    # The query returns campaign metrics
    mock_metrics_result = MagicMock()
    mock_metrics_result.mappings.return_value.all.return_value = [
        {
            "campaign_id": 1,
            "spend_7d": 100.0,
            "sales_7d": 300.0,
            "clicks_7d": 50,
            "orders_7d": 10,
            "acos_7d": 0.33,
            "roas_7d": 3.0
        },
        {
            "campaign_id": 2,
            "spend_7d": 50.0,
            "sales_7d": 100.0,
            "clicks_7d": 20,
            "orders_7d": 5,
            "acos_7d": 0.5,
            "roas_7d": 2.0
        }
    ]
    
    # Mock execute result for _load_portfolio_agent
    mock_agent_result = MagicMock()
    # first() returns None implies new agent
    mock_agent_result.mappings.return_value.first.return_value = None 
    
    # Mock execute result for _save_portfolio_state
    mock_save_state_result = MagicMock()
    mock_save_state_result.scalar.return_value = 123 # state_id
    
    # Mock execute result for CampaignAgent.allocate (keywords)
    mock_keywords_result = MagicMock()
    mock_keywords_result.mappings.return_value.all.return_value = [
        {
            "keyword_id": 101,
            "current_bid": 1.0,
            "spend_7d": 50.0, 
            "sales_7d": 150.0, # ROAS 3
            "clicks_7d": 20,
            "orders_7d": 5
        },
        {
            "keyword_id": 102,
            "current_bid": 0.5,
            "spend_7d": 10.0,
            "sales_7d": 50.0, # ROAS 5
            "clicks_7d": 5,
            "orders_7d": 2
        }
    ]

    # Chain the side_effects for db.execute
    async def execute_side_effect(statement, params=None):
        stmt_str = str(statement)
        # We check for substrings identifying the query
        if "FROM ppc_campaigns" in stmt_str or "campaign_metrics" in stmt_str:
            return mock_metrics_result
        if "rl_portfolio_state" in stmt_str and "agent_params" in stmt_str and "SELECT" in stmt_str:
            return mock_agent_result
        if "INSERT INTO rl_portfolio_state" in stmt_str and "RETURNING id" in stmt_str:
            return mock_save_state_result
        if "FROM ppc_keywords" in stmt_str:
             return mock_keywords_result
        
        # Default for INSERTs (actions, agent save, etc)
        return MagicMock()

    mock_db.execute.side_effect = execute_side_effect
    
    # Reuse mock_keywords_result for both campaigns
    # To differentiate, we could check params['cid'] but reusing is fine for dry run verification
    
    with patch("app.modules.amazon_ppc.ml.hierarchical_rl.KeywordAgent") as MockKeywordAgent:
        mock_kw_agent_instance = MockKeywordAgent.return_value
        # Mock select_bid_multiplier to return (arm_id, multiplier, expected_reward)
        mock_kw_agent_instance.select_bid_multiplier = AsyncMock(return_value=(0, 1.1, 0.5))
        
        controller = HierarchicalBudgetController(mock_db)
        
        print("\nRunning Hierarchical Allocation (Dry Run)...")
        result = await controller.run_allocation(
            profile_id="TEST_PROFILE",
            total_budget=1000.0,
            dry_run=True,
        )
        
        print(f"Result Total Budget: {result['total_budget']}")
        print(f"Num Campaigns: {result['num_campaigns']}")
        print(f"Num Keywords Allocated: {len(result['keyword_allocations'])}")

        assert result["total_budget"] == 1000.0
        assert result["num_campaigns"] == 2
        
        # We have 2 campaigns. 
        # CampaignAgent queries db twice (once per campaign).
        # Each query returns 2 keywords.
        # So total 4 keyword allocations.
        assert len(result["keyword_allocations"]) == 4 
        
        # Verify DB calls
        assert mock_db.execute.call_count >= 5
        assert mock_db.commit.called
        print("Allocation flow verified successfully.")

