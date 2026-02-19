import sys
from unittest.mock import MagicMock, AsyncMock

# Mock heavy dependencies before import
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['scikit_learn'] = MagicMock()
# Mock embedding_service module structure
mock_emb_pkg = MagicMock()
mock_emb_svc = MagicMock() # The service instance
mock_emb_pkg.embedding_service = mock_emb_svc 
sys.modules['app.services.ml.embedding_service'] = mock_emb_pkg

import unittest
from unittest.mock import patch
from app.modules.amazon_ppc.features.keyword_features import KeywordFeatureEngineer
from app.modules.amazon_ppc.models.ppc_data import PPCKeyword

class TestKeywordFeatureEngineer(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        self.mock_db = AsyncMock()
        
    async def test_compute_keyword_features(self):
        # Setup mocks
        mock_keyword = PPCKeyword(id=1, keyword_text="test", match_type="EXACT", bid=1.0, state="ENABLED")
        mock_keyword.vector = None
        
        self.mock_db.execute.side_effect = [
            MagicMock(scalars=lambda: MagicMock(first=lambda: mock_keyword)), # keyword query
            MagicMock(first=lambda: MagicMock(impressions=1000, clicks=50, spend=20.0, sales=100.0, orders=5, days_active=30)) # perf query
        ]
        
        # Mock embedding service call within method
        # We need to ensure embedding_service.encode is mocked.
        # Since we mocked the module, we need to configure that mock.
        mock_emb_svc.encode = MagicMock(return_value=MagicMock(tolist=lambda:[0.1, 0.2]))
        
        # Also need to patch run_in_executor
        with patch('asyncio.get_running_loop') as mock_loop_factory:
            mock_loop = MagicMock()
            mock_loop.run_in_executor = AsyncMock(return_value=MagicMock(tolist=lambda: [0.1, 0.2]))
            mock_loop_factory.return_value = mock_loop
            
            engineer = KeywordFeatureEngineer(self.mock_db)
            features = await engineer.compute_keyword_features(1)
            
            self.assertEqual(features['clicks'], 50)
            self.assertEqual(features['embedding'], [0.1, 0.2])

    async def test_bulk_compute_features(self):
        # Setup Mocks - use MagicMock to simulate lightweight Row objects
        # bulk_compute_features selects columns directly, so rows have .id, .keyword_text, .embedding etc.
        kw1 = MagicMock(id=1, keyword_text="k1", match_type="EXACT", state="ENABLED", bid=1.0, embedding=[0.5, 0.5])
        kw2 = MagicMock(id=2, keyword_text="k2", match_type="BROAD", state="ENABLED", bid=1.0, embedding=None)
        
        # DB Mocks
        perf_row1 = MagicMock(keyword_id=1, impressions=100, clicks=10, spend=5.0, sales=20.0, orders=1, days_active=10)
        perf_row2 = MagicMock(keyword_id=2, impressions=200, clicks=5, spend=2.0, sales=0.0, orders=0, days_active=10)

        self.mock_db.execute.side_effect = [
            MagicMock(all=lambda: [kw1, kw2]), # Keywords (bulk uses result.all(), not scalars)
            MagicMock(all=lambda: [perf_row1, perf_row2]) # Performance
        ]
        
        # Mock batch embedding
        with patch('asyncio.get_running_loop') as mock_loop_factory:
            mock_loop = MagicMock()
            # return value of run_in_executor: list of vectors (which have .tolist())
            vec_mock = MagicMock()
            vec_mock.tolist.return_value = [0.9, 0.9]
            mock_loop.run_in_executor = AsyncMock(return_value=[vec_mock])
            mock_loop_factory.return_value = mock_loop
            
            engineer = KeywordFeatureEngineer(self.mock_db)
            features = await engineer.bulk_compute_features(10)
            
            self.assertEqual(len(features), 2)
            f1 = next(f for f in features if f['keyword_id'] == 1)
            f2 = next(f for f in features if f['keyword_id'] == 2)
            
            self.assertEqual(f1['clicks'], 10)
            self.assertEqual(f1['embedding'], [0.5, 0.5]) # From Vector
            self.assertEqual(f2['clicks'], 5)
            self.assertEqual(f2['embedding'], [0.9, 0.9]) # From Batch

if __name__ == '__main__':
    unittest.main()
