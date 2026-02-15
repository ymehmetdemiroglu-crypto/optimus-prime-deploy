
import pytest
import os
import shutil
import logging
from app.modules.amazon_ppc.ml.deep_optimizer import DeepBidOptimizer
from app.modules.amazon_ppc.ml.bid_optimizer import BidOptimizer, BidPrediction
from app.modules.amazon_ppc.features.config import FeatureConfig

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_MODEL_DIR = "tests/temp_models"

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Setup
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)
    yield
    # Teardown
    if os.path.exists(TEMP_MODEL_DIR):
        shutil.rmtree(TEMP_MODEL_DIR)

class TestDeepBidOptimizerLifecycle:
    """Test the lifecycle of the PyTorch DeepBidOptimizer."""
    
    def test_full_lifecycle(self):
        model_path = os.path.join(TEMP_MODEL_DIR, "deep_bid_optimizer.pth")
        
        # 1. Initialize & Train
        model = DeepBidOptimizer(model_path=model_path)
        
        # Create Dummy Data
        training_data = []
        for i in range(100):
            sample = {f: 0.5 for f in FeatureConfig.MODEL_FEATURES}
            sample['current_bid'] = 1.0 + (i * 0.01)
            sample['optimal_bid'] = 1.2 + (i * 0.01) # Target
            training_data.append(sample)
            
        train_result = model.train(training_data, epochs=5)
        
        assert train_result['status'] == 'trained'
        assert os.path.exists(model_path), "Model file was not created"
        
        # 2. Load New Instance
        loaded_model = DeepBidOptimizer(model_path=model_path)
        assert loaded_model.is_trained, "Loaded model should be flagged as trained"
        
        # 3. Predict matches
        test_features = {f: 0.5 for f in FeatureConfig.MODEL_FEATURES}
        test_features['current_bid'] = 1.0
        
        # Model 1 prediction
        pred1, unc1 = model.predict(test_features)
        # Model 2 prediction
        pred2, unc2 = loaded_model.predict(test_features)
        
        logger.info(f"Original Pred: {pred1}, Loaded Pred: {pred2}")
        
        assert pred1 == pytest.approx(pred2, abs=1e-6), "Predictions should match between original and loaded model"
        assert pred1 > 0, "Prediction should be positive"


class TestBidOptimizerLifecycle:
    """Test the lifecycle of the Sklearn BidOptimizer."""
    
    def test_full_lifecycle(self):
        # We need to manually handle persistence for the sklearn model 
        # as it relies on external save/load (e.g. ModelStore) usually, 
        # but let's see if we can integration test the class's behavior logic.
        
        # NOTE: BidOptimizer DOES NOT fail if model artifact is None, it just uses fallback.
        # But here we want to ensure we CAN train it.
        
        optimizer = BidOptimizer()
        
        # Dummy Data
        training_data = []
        for i in range(100):
            sample = {f: 0.5 for f in FeatureConfig.MODEL_FEATURES}
            sample['current_bid'] = 1.0
            sample['optimal_bid'] = 1.2
            training_data.append(sample)
            
        result = optimizer.train(training_data)
        assert result['status'] == 'trained'
        assert result['samples'] == 100
        assert optimizer.market_model.is_trained
        
        # Test Prediction
        pred = optimizer.predict_bid(training_data[0])
        assert isinstance(pred, BidPrediction)
        assert pred.predicted_bid > 0
