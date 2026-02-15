"""
LSTM-based Time Series Forecaster for PPC Metrics.
Captures sequential patterns and long-term dependencies.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class LSTMCell:
    """Single LSTM cell implementation."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrices for efficiency
        combined_size = input_size + hidden_size
        
        # Forget gate
        self.Wf = np.random.randn(combined_size, hidden_size) * 0.1
        self.bf = np.ones(hidden_size)  # Initialize to 1 for stable training
        
        # Input gate
        self.Wi = np.random.randn(combined_size, hidden_size) * 0.1
        self.bi = np.zeros(hidden_size)
        
        # Candidate values
        self.Wc = np.random.randn(combined_size, hidden_size) * 0.1
        self.bc = np.zeros(hidden_size)
        
        # Output gate
        self.Wo = np.random.randn(combined_size, hidden_size) * 0.1
        self.bo = np.zeros(hidden_size)
    
    def forward(
        self, 
        x: np.ndarray, 
        h_prev: np.ndarray, 
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM cell."""
        
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev], axis=-1)
        
        # Forget gate
        f = self._sigmoid(combined @ self.Wf + self.bf)
        
        # Input gate
        i = self._sigmoid(combined @ self.Wi + self.bi)
        
        # Candidate values
        c_tilde = np.tanh(combined @ self.Wc + self.bc)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = self._sigmoid(combined @ self.Wo + self.bo)
        
        # New hidden state
        h = o * np.tanh(c)
        
        return h, c
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTMForecaster:
    """
    LSTM-based forecaster for PPC time series.
    Predicts multiple metrics simultaneously.
    """
    
    METRICS = ['impressions', 'clicks', 'spend', 'sales', 'orders']
    
    def __init__(
        self, 
        sequence_length: int = 14,
        hidden_size: int = 32,
        model_path: Optional[str] = None
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.model_path = model_path or "models/lstm_forecaster.pkl"
        
        self.n_features = len(self.METRICS)
        
        # LSTM layers
        self.lstm1 = LSTMCell(self.n_features, hidden_size)
        self.lstm2 = LSTMCell(hidden_size, hidden_size)
        
        # Output layer
        self.Wy = np.random.randn(hidden_size, self.n_features) * 0.1
        self.by = np.zeros(self.n_features)
        
        # Normalization parameters
        self.means = None
        self.stds = None
        
        self._load_model()
    
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.lstm1 = data['lstm1']
                    self.lstm2 = data['lstm2']
                    self.Wy = data['Wy']
                    self.by = data['by']
                    self.means = data.get('means')
                    self.stds = data.get('stds')
                logger.info(f"Loaded LSTM forecaster from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
    
    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'lstm1': self.lstm1,
                'lstm2': self.lstm2,
                'Wy': self.Wy,
                'by': self.by,
                'means': self.means,
                'stds': self.stds
            }, f)
    
    def _prepare_sequence(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Convert list of daily records to numpy array."""
        sequence = []
        for record in data:
            row = [float(record.get(m, 0)) for m in self.METRICS]
            sequence.append(row)
        return np.array(sequence)
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self.means is None:
            self.means = data.mean(axis=0)
            self.stds = data.std(axis=0) + 1e-8
        return (data - self.means) / self.stds
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        if self.means is None:
            return data
        return data * self.stds + self.means
    
    def forecast(
        self, 
        historical_data: List[Dict[str, Any]], 
        horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Generate multi-step forecast.
        
        Args:
            historical_data: List of daily performance records
            horizon: Number of days to forecast
        
        Returns:
            Dictionary with forecasts for each metric
        """
        if len(historical_data) < self.sequence_length:
            # Pad with last value
            while len(historical_data) < self.sequence_length:
                historical_data.insert(0, historical_data[0])
        
        # Prepare and normalize input
        sequence = self._prepare_sequence(historical_data[-self.sequence_length:])
        sequence = self._normalize(sequence)
        
        forecasts = []
        
        # Initialize hidden states
        h1 = np.zeros(self.hidden_size)
        c1 = np.zeros(self.hidden_size)
        h2 = np.zeros(self.hidden_size)
        c2 = np.zeros(self.hidden_size)
        
        # Process sequence
        for t in range(self.sequence_length):
            h1, c1 = self.lstm1.forward(sequence[t], h1, c1)
            h2, c2 = self.lstm2.forward(h1, h2, c2)
        
        # Generate forecasts
        current_input = sequence[-1]
        
        for _ in range(horizon):
            h1, c1 = self.lstm1.forward(current_input, h1, c1)
            h2, c2 = self.lstm2.forward(h1, h2, c2)
            
            # Output
            output = h2 @ self.Wy + self.by
            forecasts.append(output)
            
            # Use output as next input
            current_input = output
        
        # Denormalize forecasts
        forecasts = np.array(forecasts)
        forecasts = self._denormalize(forecasts)
        
        # Ensure non-negative values
        forecasts = np.maximum(0, forecasts)
        
        # Build result
        result = {'horizon': horizon, 'metrics': {}}
        
        for i, metric in enumerate(self.METRICS):
            result['metrics'][metric] = {
                'forecast': forecasts[:, i].tolist(),
                'trend': self._detect_trend(forecasts[:, i])
            }
        
        # Calculate derived metrics
        if 'spend' in result['metrics'] and 'sales' in result['metrics']:
            spend_forecast = forecasts[:, self.METRICS.index('spend')]
            sales_forecast = forecasts[:, self.METRICS.index('sales')]
            
            acos_forecast = np.where(
                sales_forecast > 0,
                spend_forecast / sales_forecast * 100,
                0
            )
            
            result['metrics']['acos'] = {
                'forecast': acos_forecast.tolist(),
                'trend': self._detect_trend(acos_forecast)
            }
        
        return result
    
    def _detect_trend(self, values: np.ndarray) -> str:
        if len(values) < 2:
            return 'stable'
        slope = np.polyfit(range(len(values)), values, 1)[0]
        mean_val = np.mean(values)
        if mean_val == 0:
            return 'stable'
        relative_slope = slope / mean_val
        if relative_slope > 0.02:
            return 'up'
        elif relative_slope < -0.02:
            return 'down'
        return 'stable'
    
    def train(
        self, 
        training_data: List[List[Dict[str, Any]]],
        epochs: int = 50,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train the LSTM on historical data.
        
        Args:
            training_data: List of sequences (each sequence is a list of daily records)
        """
        if len(training_data) < 10:
            return {'status': 'insufficient_data'}
        
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for sequence_data in training_data:
                if len(sequence_data) < self.sequence_length + 1:
                    continue
                
                sequence = self._prepare_sequence(sequence_data)
                sequence = self._normalize(sequence)
                
                X = sequence[:-1]
                y = sequence[1:]
                
                # Forward pass
                h1 = np.zeros(self.hidden_size)
                c1 = np.zeros(self.hidden_size)
                h2 = np.zeros(self.hidden_size)
                c2 = np.zeros(self.hidden_size)
                
                for t in range(len(X)):
                    h1, c1 = self.lstm1.forward(X[t], h1, c1)
                    h2, c2 = self.lstm2.forward(h1, h2, c2)
                    
                    output = h2 @ self.Wy + self.by
                    loss = np.mean((output - y[t]) ** 2)
                    epoch_loss += loss
                    
                    # Simplified gradient update
                    grad = 2 * (output - y[t]) / len(y[t])
                    self.Wy -= learning_rate * np.outer(h2, grad)
                    self.by -= learning_rate * grad
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
            
            total_loss = epoch_loss
        
        self._save_model()
        
        return {
            'status': 'trained',
            'epochs': epochs,
            'final_loss': total_loss
        }


class SeasonalDecomposer:
    """
    Decomposes time series into trend, seasonality, and residual.
    Useful for understanding underlying patterns.
    """
    
    def __init__(self, period: int = 7):
        self.period = period
    
    def decompose(self, values: List[float]) -> Dict[str, List[float]]:
        """
        Decompose time series using moving average.
        """
        values = np.array(values)
        n = len(values)
        
        if n < self.period * 2:
            return {
                'original': values.tolist(),
                'trend': values.tolist(),
                'seasonal': [0] * n,
                'residual': [0] * n
            }
        
        # Calculate trend using centered moving average
        trend = np.convolve(values, np.ones(self.period) / self.period, mode='valid')
        
        # Pad trend to match original length
        pad_left = (n - len(trend)) // 2
        pad_right = n - len(trend) - pad_left
        trend = np.concatenate([
            np.full(pad_left, trend[0]),
            trend,
            np.full(pad_right, trend[-1])
        ])
        
        # Calculate detrended series
        detrended = values - trend
        
        # Calculate seasonal component (average for each day of week)
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = range(i, n, self.period)
            seasonal_avg = np.mean([detrended[j] for j in indices])
            for j in indices:
                seasonal[j] = seasonal_avg
        
        # Normalize seasonal component
        seasonal = seasonal - np.mean(seasonal)
        
        # Calculate residual
        residual = values - trend - seasonal
        
        return {
            'original': values.tolist(),
            'trend': trend.tolist(),
            'seasonal': seasonal.tolist(),
            'residual': residual.tolist(),
            'period': self.period
        }
    
    def get_seasonal_indices(self, values: List[float]) -> List[float]:
        """Get seasonal indices for each day of the period."""
        decomposition = self.decompose(values)
        seasonal = decomposition['seasonal']
        
        # Average by position in period
        indices = []
        for i in range(self.period):
            avg = np.mean([seasonal[j] for j in range(i, len(seasonal), self.period)])
            indices.append(avg)
        
        return indices
