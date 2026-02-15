import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class LSTMPriceModel(nn.Module):
    """
    LSTM model for predicting future price movements.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=7, dropout=0.2):
        super(LSTMPriceModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class Forecaster:
    """
    Wrapper for training and inference.
    """
    def __init__(self, sequence_length=30, forecast_horizon=7):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMPriceModel(output_dim=forecast_horizon).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train(self, historical_prices: List[float], epochs=100):
        """Train model on a single price series (simplified for MVP)."""
        if len(historical_prices) < self.sequence_length + self.forecast_horizon:
            logger.warning("Insufficient data for training")
            return

        # Prepare data
        sequences = []
        targets = []
        
        # Sliding window
        for i in range(len(historical_prices) - self.sequence_length - self.forecast_horizon):
            seq = historical_prices[i : i+self.sequence_length]
            target = historical_prices[i+self.sequence_length : i+self.sequence_length+self.forecast_horizon]
            sequences.append(seq)
            targets.append(target)
            
        if not sequences:
            return

        X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1).to(self.device) # (N, 30, 1)
        y = torch.tensor(targets, dtype=torch.float32).to(self.device)                 # (N, 7)
        
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, recent_prices: List[float]) -> List[float]:
        """Generate forecast for next N days."""
        if len(recent_prices) < self.sequence_length:
            # Pad with last value if short
            padding = [recent_prices[-1]] * (self.sequence_length - len(recent_prices))
            recent_prices = padding + recent_prices
            
        recent_prices = recent_prices[-self.sequence_length:] # Take last 30
        
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor([recent_prices], dtype=torch.float32).unsqueeze(-1).to(self.device)
            prediction = self.model(X)
            return prediction.cpu().numpy()[0].tolist()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
