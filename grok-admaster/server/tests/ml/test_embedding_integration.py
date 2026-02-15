
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.services.ml.embedding_service import EmbeddingService
from app.modules.amazon_ppc.ml.deep_optimizer import DeepBidOptimizer

@pytest.mark.asyncio
async def test_embedding_service():
    # Mock SentenceTransformer to avoid downloading model during tests
    with patch('app.services.ml.embedding_service.SentenceTransformer') as MockTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        MockTransformer.return_value = mock_model
        
        service = EmbeddingService()
        # Reset singleton for test
        EmbeddingService._instance = service
        
        vec = service.encode("test keyword")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)

@pytest.mark.asyncio
async def test_optimizer_integration():
    # Test that optimizer handles the embedding input correctly
    optimizer = DeepBidOptimizer()
    
    # Mock data with embedding
    feature_data = {
        'current_bid': 1.0,
        'ctr_30d': 0.05,
        'embedding': np.random.rand(384).astype(np.float32)
    }
    
    # 1. Test Feature Preparation
    input_vec = optimizer._prepare_features(feature_data)
    expected_size = len(optimizer.FEATURE_COLS) + 384
    assert input_vec.shape == (expected_size,)
    
    # 2. Test Normalization logic
    # Create a batch of 2
    import torch
    X = torch.tensor(np.array([input_vec, input_vec]), dtype=torch.float32).to(optimizer.device)
    
    # Mock means/stds
    scalar_len = len(optimizer.FEATURE_COLS)
    optimizer.feature_means = np.zeros(scalar_len)
    optimizer.feature_stds = np.ones(scalar_len)
    
    X_norm = optimizer._normalize(X)
    assert X_norm.shape == (2, expected_size)
    
    # 3. Test Prediction flow (forward pass)
    bid, uncertainty = optimizer.predict(feature_data)
    assert isinstance(bid, float)
    assert bid > 0

