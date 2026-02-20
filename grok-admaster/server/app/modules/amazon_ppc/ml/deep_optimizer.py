"""
Neural Network-based Bid Optimizer using PyTorch.
Implements a deep learning approach for bid prediction.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from app.modules.amazon_ppc.features.config import FeatureConfig

logger = logging.getLogger(__name__)

class BidNetwork(nn.Module):
    """
    PyTorch Neural Network for bid optimization.
    Architecture: Input -> 128 -> 64 -> 32 -> 16 -> 1
    """
    def __init__(self, input_size: int):
        super(BidNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 1)
        )
        
        # Initialize weights specifically if needed, though PyTorch defaults are usually fine
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)

class DeepBidOptimizer:
    """
    Deep Neural Network for bid optimization using PyTorch.
    """
    
    FEATURE_COLS = FeatureConfig.MODEL_FEATURES
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/deep_bid_optimizer.pth"
        # Base features + Embedding dim (384)
        self.embedding_dim = 384
        self.input_size = len(self.FEATURE_COLS) + self.embedding_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BidNetwork(self.input_size).to(self.device)
        self.is_trained = False
        
        # Feature normalization stats (only for scalar features)
        self.feature_means = None
        self.feature_stds = None
        
        self._load_model()
    
    def _save_model(self):
        """Save trained model to disk."""
        try:
            os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'means': self.feature_means,
                'stds': self.feature_stds,
                'date': str(datetime.now()),
            }
            torch.save(checkpoint, self.model_path)
            logger.info(f"Saved deep bid optimizer to {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                # Check for architecture mismatch (e.g. if loaded model was trained without embeddings)
                saved_input_size = checkpoint['model_state_dict']['network.0.weight'].shape[1]
                if saved_input_size != self.input_size:
                    logger.warning(f"Model architecture mismatch: Saved {saved_input_size}, Current {self.input_size}. Re-initializing model.")
                    return

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.feature_means = checkpoint['means']
                self.feature_stds = checkpoint['stds']
                self.is_trained = True
                self.model.eval()
                logger.info(f"Loaded deep bid optimizer from {self.model_path} (trained on {checkpoint.get('date', 'unknown')})")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

    def _prepare_features(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        # 1. Scalar Features
        scalar_values = []
        for col in self.FEATURE_COLS:
            val = feature_dict.get(col, 0)
            if isinstance(val, bool):
                val = int(val)
            scalar_values.append(float(val) if val is not None else 0.0)
        
        # 2. Embedding Features
        embedding = feature_dict.get('embedding')
        if embedding is None:
            # Zero vector if missing
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        elif isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Concatenate
        return np.concatenate([np.array(scalar_values, dtype=np.float32), embedding])

    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        if self.feature_means is None:
            return X
            
        # Only normalize the scalar features part
        scalar_count = len(self.FEATURE_COLS)
        
        # Split
        X_scalar = X[:, :scalar_count]
        X_embedding = X[:, scalar_count:]
        
        means = torch.tensor(self.feature_means, dtype=torch.float32, device=self.device)
        stds = torch.tensor(self.feature_stds, dtype=torch.float32, device=self.device)
        
        # Normalize scalars
        X_scalar_norm = (X_scalar - means) / (stds + 1e-8)
        
        # Re-concatenate
        return torch.cat([X_scalar_norm, X_embedding], dim=1)

    def train(
        self,
        training_data: List[Dict[str, Any]],
        target_col: str = 'optimal_bid',
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train the neural network using PyTorch."""
        
        if len(training_data) < 50:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return {'status': 'insufficient_data'}
        
        self.model.train()
        
        # Prepare data
        X_numpy = np.array([self._prepare_features(d) for d in training_data])
        y_numpy = np.array([d.get(target_col, d.get('current_bid', 1.0)) for d in training_data], dtype=np.float32).reshape(-1, 1)
        
        # Calculate normalization stats (only for scalar columns)
        scalar_count = len(self.FEATURE_COLS)
        self.feature_means = X_numpy[:, :scalar_count].mean(axis=0)
        self.feature_stds = X_numpy[:, :scalar_count].std(axis=0)
        
        # Convert to tensors
        X = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_numpy, dtype=torch.float32).to(self.device)
        
        # Normalize
        X = self._normalize(X)
        
        # Optimizer & Loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        dataset_size = len(X)
        history = {'loss': []}
        
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            batches = 0
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], y[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
                
            avg_loss = epoch_loss / batches
            history['loss'].append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        self.is_trained = True
        self.model.eval()
        self._save_model()
        
        return {
            'status': 'trained',
            'epochs': epochs,
            'final_loss': history['loss'][-1],
            'samples': len(training_data),
            'device': str(self.device)
        }

    def predict(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict optimal bid.
        Returns: (predicted_bid, uncertainty)
        """
        self.model.eval()
        
        X_numpy = self._prepare_features(features)
        X = torch.tensor(X_numpy, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.feature_means is not None:
                X = self._normalize(X)
            
            prediction = self.model(X).item()
        
        # Ensure positive bid
        predicted_bid = max(0.10, prediction)
        
        # Estimate uncertainty based on data maturity
        data_maturity = features.get('data_maturity', 0.5)
        uncertainty = 0.3 * (1 - data_maturity)
        
        return predicted_bid, uncertainty
