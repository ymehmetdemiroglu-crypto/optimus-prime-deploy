
import sys
import os
import logging
import time

# Add server root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from app.modules.amazon_ppc.ml.rl_agent import PPCRLAgent

logging.basicConfig(level=logging.INFO)

def test_dqn_agent():
    print("Initializing DQN Agent...")
    agent = PPCRLAgent(model_path="models/test_dqn_agent.pth")
    
    # Dummy history
    history = [
        {
            'before_features': {'acos_7d': 30.0, 'momentum': 0.1, 'spend_trend': 1.0, 'sales': 100},
            'after_features': {'acos_7d': 25.0, 'momentum': 0.2, 'spend_trend': 1.1, 'sales': 110},
            'bid_change': 1.05 # Increased 5%
        }
        for _ in range(50)
    ]
    
    print("Training from history...")
    try:
        result = agent.train_from_history(history)
        print("Training result:", result)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Getting recommendation...")
    try:
        features = {'acos_7d': 28.0, 'momentum': 0.05, 'current_bid': 1.5}
        rec = agent.get_bid_recommendation(features, current_bid=1.5)
        print(f"Recommendation: {rec}")
    except Exception as e:
        print(f"Recommendation failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    try:
        if test_dqn_agent():
            print("VERIFICATION SUCCESS: DQN Agent works!")
        else:
            print("VERIFICATION FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)
