"""
Reinforcement Learning Agent for PPC Bid Optimization.
Uses Deep Q-Learning (DQN) for real-time bid adjustments.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import os
import logging

from .dqn import DQN, ReplayBuffer

logger = logging.getLogger(__name__)

@dataclass
class RLAction:
    """Action the agent can take."""
    action_id: int
    bid_multiplier: float
    description: str

class PPCRLAgent:
    """
    Deep Q-Learning agent for PPC bid optimization.
    """
    
    ACTIONS = [
        RLAction(0, 0.80, "Decrease 20%"),
        RLAction(1, 0.90, "Decrease 10%"),
        RLAction(2, 0.95, "Decrease 5%"),
        RLAction(3, 1.00, "Maintain"),
        RLAction(4, 1.05, "Increase 5%"),
        RLAction(5, 1.10, "Increase 10%"),
        RLAction(6, 1.20, "Increase 20%"),
    ]
    
    # State vector dimensions (e.g., ACoS ratio, momentum, etc.)
    STATE_SIZE = 6 
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        model_path: Optional[str] = None
    ):
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.model_path = model_path or "models/rl_agent_dqn.pth"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN Networks
        self.policy_net = DQN(self.STATE_SIZE, len(self.ACTIONS)).to(self.device)
        self.target_net = DQN(self.STATE_SIZE, len(self.ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.update_target_every = 100
        self.steps_done = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if exists."""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Loaded DQN agent from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load DQN model: {e}")
    
    def save_model(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path)
        logger.info(f"Saved DQN agent to {self.model_path}")

    def _get_state_vector(self, features: Dict[str, Any], target_acos: float = 25.0) -> np.ndarray:
        """Convert features to continuous state vector."""
        # 1. ACoS Ratio
        acos = features.get('acos_30d', features.get('acos_7d', 50))
        acos_ratio = acos / (target_acos + 1e-6)
        
        # 2. Momentum (Trend)
        momentum = features.get('momentum', 0.0)
        
        # 3. Spend Trend (Budget Pacing)
        spend_trend = features.get('spend_trend', 1.0)
        
        # 4. CPC Volatility (Competition)
        cpc_volatility = features.get('cpc_volatility', 0.0)
        
        # 5. Conversion Rate (normalized approx)
        cvr = features.get('conversion_rate_7d', 0.0) * 10 
        
        # 6. Current Bid (normalized approx, assuming $1-$5 range)
        bid = features.get('current_bid', 1.0) / 2.0
        
        return np.array([acos_ratio, momentum, spend_trend, cpc_volatility, cvr, bid], dtype=np.float32)

    def select_action(self, features: Dict[str, Any], current_bid: float, target_acos: float = 25.0, explore: bool = True) -> RLAction:
        """Select action using epsilon-greedy policy with DQN."""
        state = self._get_state_vector(features, target_acos)
        
        if explore and np.random.random() < self.epsilon:
            action_id = np.random.randint(len(self.ACTIONS))
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                action_id = q_values.argmax().item()
                
        return self.ACTIONS[action_id]

    def _train_step(self):
        """Perform one step of optimization on the Policy Net."""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Q(s, a)
        current_q_values = self.policy_net(states_t).gather(1, actions_t)
        
        # V(s') = max Q(s', a')
        next_q_values = self.target_net(next_states_t).max(1)[0].detach()
        expected_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))
        
        loss = self.criterion(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self, features: Dict[str, Any], action: RLAction, reward: float, next_features: Dict[str, Any], done: bool = False):
        """Add experience and train."""
        state = self._get_state_vector(features)
        next_state = self._get_state_vector(next_features)
        
        self.memory.push(state, action.action_id, reward, next_state, done)
        self._train_step()

    def calculate_reward(self, prev_features: Dict[str, Any], curr_features: Dict[str, Any], target_acos: float = 25.0) -> float:
        """Same reward function as before."""
        reward = 0.0
        prev_acos = prev_features.get('acos_7d', 50)
        curr_acos = curr_features.get('acos_7d', 50)
        prev_sales = prev_features.get('sales', 0)
        curr_sales = curr_features.get('sales', 0)
        
        # ACoS target achievement
        if 0.8 * target_acos <= curr_acos <= 1.2 * target_acos:
            reward += 1.0
        elif curr_acos > target_acos * 1.5:
            reward -= 0.5
        
        # ACoS improvement
        if curr_acos < prev_acos:
            reward += 0.3
        elif curr_acos > prev_acos * 1.1:
            reward -= 0.2
        
        # Sales improvement
        if prev_sales > 0:
            sales_change = (curr_sales - prev_sales) / prev_sales
            if sales_change > 0.1:
                reward += 0.5
            elif sales_change < -0.1:
                reward -= 0.5
        return reward

    def get_bid_recommendation(
        self,
        features: Dict[str, Any],
        current_bid: float,
        target_acos: float = 25.0
    ) -> Dict[str, Any]:
        """Public inference method."""
        action = self.select_action(features, current_bid, target_acos, explore=False)
        
        new_bid = current_bid * action.bid_multiplier
        new_bid = max(0.10, min(new_bid, current_bid * 1.5))
        
        return {
            'current_bid': current_bid,
            'recommended_bid': round(new_bid, 2),
            'action': action.description,
            'confidence': "high", # DQN always outputs a value
            'model_type': 'DQN'
        }

    def train_from_history(self, history: List[Dict[str, Any]], target_acos: float = 25.0) -> Dict[str, Any]:
        """Train from batch history."""
        total_loss = 0
        episodes = 0
        
        for record in history:
            before = record.get('before_features', {})
            after = record.get('after_features', {})
            bid_change = record.get('bid_change', 1.0)
            
            # Find action
            action_obj = min(self.ACTIONS, key=lambda a: abs(a.bid_multiplier - bid_change))
            reward = self.calculate_reward(before, after, target_acos)
            
            self.update(before, action_obj, reward, after, done=False)
            episodes += 1
            
        self.save_model()
        
        return {
            'episodes': episodes,
            'status': 'trained_dqn'
        }
