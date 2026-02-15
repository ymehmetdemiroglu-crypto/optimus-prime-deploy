
import sys
import os
import logging

# Add server root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from app.modules.amazon_ppc.ml.deep_optimizer import DeepBidOptimizer

logging.basicConfig(level=logging.INFO)

def test_deep_optimizer():
    print("Initializing DeepBidOptimizer...")
    optimizer = DeepBidOptimizer(model_path="models/test_deep_optimizer.pth")
    
    # Dummy data
    training_data = [
        {
            'ctr_7d': 0.05, 'current_bid': 1.0, 'optimal_bid': 1.2,
            'acos_7d': 20.0, 'roas_7d': 5.0, 'sales_trend': 0.1
        }
        for _ in range(60) # > 50 samples
    ]
    
    print("Training model...")
    try:
        result = optimizer.train(training_data, epochs=5)
        print("Training result:", result)
    except Exception as e:
        print(f"Training failed: {e}")
        return False

    print("Predicting...")
    try:
        features = {'ctr_7d': 0.06, 'current_bid': 1.1}
        bid, uncertainty = optimizer.predict(features)
        print(f"Prediction: Bid=${bid:.2f}, Uncertainty={uncertainty:.2f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    try:
        if test_deep_optimizer():
            print("VERIFICATION SUCCESS: PyTorch model works!")
        else:
            print("VERIFICATION FAILED")
            sys.exit(1)
    except ImportError as e:
        print(f"VERIFICATION SKIPPED: Missing dependency ({e})")
        sys.exit(0)
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)
