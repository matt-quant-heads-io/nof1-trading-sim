import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import gym
from gym import spaces
from collections import deque
import logging

# Action space definition
class TradingAction(IntEnum):
    DO_NOTHING = 0
    PLACE_PASSIVE_BUY_ORDER = 1
    PLACE_PASSIVE_SELL_ORDER = 2
    CANCEL_PASSIVE_ORDERS = 3
    ATTEMPT_MARKET_BUY = 4
    ATTEMPT_MARKET_SELL = 5
    EXIT_ALL = 6

@dataclass
class Order:
    """Represents a trading order"""
    id: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'limit' or 'market'
    quantity: float
    price: Optional[float]
    timestamp: int
    expiry_time: Optional[int] = None
    filled_quantity: float = 0.0
    status: str = 'pending'  # 'pending', 'filled', 'cancelled', 'expired'

@dataclass
class Position:
    """Represents current trading position"""
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class AttentionLayer(nn.Module):
    """Multi-head attention layer for LOB data"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(x + attention_output)
        return output

class LSTMAttentionModel(nn.Module):
    """LSTM with attention mechanism for LOB trading"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        action_dim: int = 7
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_lstm_layers, 
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention layers (using enhanced version for visualization)
        from attention_visualization import AttentionLayerWithWeights
        self.attention1 = AttentionLayerWithWeights(hidden_dim, num_attention_heads, dropout)
        self.attention2 = AttentionLayerWithWeights(hidden_dim, num_attention_heads, dropout)
        
        # Output layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Action value head (Q-values for each action)
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # State value head (for advantage calculation)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Optional hidden state for LSTM
        Returns:
            action_values: Q-values for each action
            state_value: State value estimate
            hidden: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention layers
        attended_1 = self.attention1(lstm_out)
        attended_2 = self.attention2(attended_1)
        
        # Use the last time step for action prediction
        final_hidden = attended_2[:, -1, :]  # (batch_size, hidden_dim)
        
        # Extract features
        features = self.feature_extractor(final_hidden)
        
        # Compute action values and state value
        action_values = self.action_head(features)
        state_value = self.value_head(features)
        
        return action_values, state_value, hidden

class LOBTradingEnvironment(gym.Env):
    """
    Limit Order Book Trading Environment
    Handles order management and position tracking
    """
    
    def __init__(
        self,
        lob_data: np.ndarray,
        lookback_window: int = 100,
        max_position_size: float = 10.0,
        transaction_cost: float = 0.001,
        order_expiry_ticks: int = 10,
        max_order_size_pct: float = 0.01,
        initial_capital: float = 100000.0
    ):
        super().__init__()
        
        self.lob_data = lob_data  # Shape: (num_ticks, num_features)
        self.num_ticks, self.num_features = lob_data.shape
        self.lookback_window = lookback_window
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.order_expiry_ticks = order_expiry_ticks
        self.max_order_size_pct = max_order_size_pct
        self.initial_capital = initial_capital
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(TradingAction))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(lookback_window, self.num_features + 10),  # +10 for position/order features
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_tick = self.lookback_window
        self.position = Position()
        self.cash = self.initial_capital
        self.active_orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step"""
        # Store previous state for reward calculation
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_action(TradingAction(action))
        
        # Update time
        self.current_tick += 1
        
        # Process existing orders (fill/expire)
        self._process_orders()
        
        # Update position PnL
        self._update_position_pnl()
        
        # Calculate reward
        current_portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        
        # Check if episode is done
        done = self.current_tick >= self.num_ticks - 1
        
        # Create info dictionary
        info = {
            'portfolio_value': current_portfolio_value,
            'position_quantity': self.position.quantity,
            'cash': self.cash,
            'active_orders': len(self.active_orders),
            'total_trades': self.total_trades,
            'unrealized_pnl': self.position.unrealized_pnl,
            'realized_pnl': self.position.realized_pnl
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: TradingAction):
        """Execute the specified trading action"""
        current_price = self._get_mid_price()
        
        if action == TradingAction.DO_NOTHING:
            pass
            
        elif action == TradingAction.PLACE_PASSIVE_BUY_ORDER:
            self._place_passive_order('buy', current_price)
            
        elif action == TradingAction.PLACE_PASSIVE_SELL_ORDER:
            self._place_passive_order('sell', current_price)
            
        elif action == TradingAction.CANCEL_PASSIVE_ORDERS:
            self._cancel_all_orders()
            
        elif action == TradingAction.ATTEMPT_MARKET_BUY:
            self._execute_market_order('buy')
            
        elif action == TradingAction.ATTEMPT_MARKET_SELL:
            self._execute_market_order('sell')
            
        elif action == TradingAction.EXIT_ALL:
            self._exit_all_positions()
    
    def _place_passive_order(self, side: str, reference_price: float):
        """Place a passive limit order"""
        # Cancel existing orders on the same side
        orders_to_cancel = [
            order_id for order_id, order in self.active_orders.items()
            if order.side == side and order.order_type == 'limit'
        ]
        for order_id in orders_to_cancel:
            del self.active_orders[order_id]
        
        # Calculate order price (slightly ahead of current best)
        bid_price, ask_price = self._get_bid_ask()
        
        if side == 'buy':
            order_price = bid_price + 0.01  # Just ahead of current best bid
        else:
            order_price = ask_price - 0.01  # Just ahead of current best ask
        
        # Calculate order size
        order_size = self._calculate_order_size()
        
        # Create order
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        
        order = Order(
            id=order_id,
            side=side,
            order_type='limit',
            quantity=order_size,
            price=order_price,
            timestamp=self.current_tick,
            expiry_time=self.current_tick + self.order_expiry_ticks
        )
        
        self.active_orders[order_id] = order
    
    def _execute_market_order(self, side: str):
        """Execute a market order"""
        order_size = self._calculate_order_size()
        current_price = self._get_mid_price()
        
        # Apply slippage and transaction costs
        if side == 'buy':
            execution_price = current_price * (1 + self.transaction_cost)
            if self.cash >= order_size * execution_price:
                self._fill_order(side, order_size, execution_price)
        else:
            execution_price = current_price * (1 - self.transaction_cost)
            if abs(self.position.quantity) >= order_size:
                self._fill_order(side, order_size, execution_price)
    
    def _exit_all_positions(self):
        """Close all positions immediately"""
        if abs(self.position.quantity) > 0:
            side = 'sell' if self.position.quantity > 0 else 'buy'
            quantity = abs(self.position.quantity)
            current_price = self._get_mid_price()
            execution_price = current_price * (1 - self.transaction_cost if side == 'sell' else 1 + self.transaction_cost)
            
            self._fill_order(side, quantity, execution_price)
        
        # Cancel all pending orders
        self._cancel_all_orders()
    
    def _cancel_all_orders(self):
        """Cancel all active orders"""
        self.active_orders.clear()
    
    def _process_orders(self):
        """Process active orders for fills and expiry"""
        current_price = self._get_mid_price()
        bid_price, ask_price = self._get_bid_ask()
        
        orders_to_remove = []
        
        for order_id, order in self.active_orders.items():
            # Check for expiry
            if order.expiry_time and self.current_tick >= order.expiry_time:
                orders_to_remove.append(order_id)
                continue
            
            # Check for fills
            if order.order_type == 'limit':
                should_fill = False
                
                if order.side == 'buy' and ask_price <= order.price:
                    should_fill = True
                elif order.side == 'sell' and bid_price >= order.price:
                    should_fill = True
                
                if should_fill:
                    self._fill_order(order.side, order.quantity, order.price)
                    orders_to_remove.append(order_id)
        
        # Remove filled/expired orders
        for order_id in orders_to_remove:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
    
    def _fill_order(self, side: str, quantity: float, price: float):
        """Fill an order and update position"""
        if side == 'buy':
            cost = quantity * price
            if self.cash >= cost:
                self.cash -= cost
                new_quantity = self.position.quantity + quantity
                if self.position.quantity == 0:
                    self.position.avg_price = price
                else:
                    total_cost = (self.position.quantity * self.position.avg_price) + cost
                    self.position.avg_price = total_cost / new_quantity
                self.position.quantity = new_quantity
                self.total_trades += 1
        else:  # sell
            if self.position.quantity >= quantity:
                revenue = quantity * price
                self.cash += revenue
                
                # Calculate realized PnL
                realized_pnl = quantity * (price - self.position.avg_price)
                self.position.realized_pnl += realized_pnl
                self.total_pnl += realized_pnl
                
                self.position.quantity -= quantity
                self.total_trades += 1
                
                if self.position.quantity == 0:
                    self.position.avg_price = 0.0
    
    def _calculate_order_size(self) -> float:
        """Calculate appropriate order size"""
        current_price = self._get_mid_price()
        volume_at_level = self._get_volume_at_level()
        
        # Order size as percentage of available liquidity
        max_size_by_liquidity = volume_at_level * self.max_order_size_pct
        
        # Order size constrained by position limits
        max_size_by_position = self.max_position_size - abs(self.position.quantity)
        
        # Order size constrained by available cash
        max_size_by_cash = self.cash / current_price * 0.95  # Leave some buffer
        
        return min(max_size_by_liquidity, max_size_by_position, max_size_by_cash, 1.0)
    
    def _update_position_pnl(self):
        """Update unrealized PnL for current position"""
        if self.position.quantity != 0:
            current_price = self._get_mid_price()
            self.position.unrealized_pnl = self.position.quantity * (current_price - self.position.avg_price)
    
    def _calculate_reward(self, prev_portfolio_value: float, current_portfolio_value: float) -> float:
        """Calculate reward for the current step"""
        # Portfolio value change
        pnl_reward = current_portfolio_value - prev_portfolio_value
        
        # Small penalty for excessive trading
        trading_penalty = -0.01 * len(self.active_orders)
        
        # Risk penalty for large positions
        position_penalty = -0.001 * abs(self.position.quantity) ** 2
        
        return pnl_reward + trading_penalty + position_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        # LOB features for lookback window
        start_idx = max(0, self.current_tick - self.lookback_window)
        end_idx = self.current_tick
        
        lob_features = self.lob_data[start_idx:end_idx]
        
        # Pad if necessary
        if lob_features.shape[0] < self.lookback_window:
            padding = np.zeros((self.lookback_window - lob_features.shape[0], self.num_features))
            lob_features = np.vstack([padding, lob_features])
        
        # Position and order features
        position_features = np.array([
            self.position.quantity / self.max_position_size,  # Normalized position
            self.position.unrealized_pnl / self.initial_capital,  # Normalized unrealized PnL
            self.position.realized_pnl / self.initial_capital,   # Normalized realized PnL
            self.cash / self.initial_capital,  # Normalized cash
            len(self.active_orders) / 10.0,   # Normalized number of active orders
            self.total_trades / 100.0,        # Normalized trade count
            0.0, 0.0, 0.0, 0.0  # Reserved for additional features
        ])
        
        # Repeat position features for each time step
        position_features_expanded = np.tile(position_features, (self.lookback_window, 1))
        
        # Combine LOB and position features
        observation = np.hstack([lob_features, position_features_expanded])
        
        return observation.astype(np.float32)
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return self.cash + self.position.quantity * self._get_mid_price()
    
    def _get_mid_price(self) -> float:
        """Get current mid price from LOB data"""
        # Assuming first two columns are best bid and ask prices
        bid_price = self.lob_data[self.current_tick, 0]
        ask_price = self.lob_data[self.current_tick, 1]
        return (bid_price + ask_price) / 2.0
    
    def _get_bid_ask(self) -> Tuple[float, float]:
        """Get current bid and ask prices"""
        bid_price = self.lob_data[self.current_tick, 0]
        ask_price = self.lob_data[self.current_tick, 1]
        return bid_price, ask_price
    
    def _get_volume_at_level(self) -> float:
        """Get volume at best level"""
        # Assuming columns 2 and 3 are bid and ask volumes
        bid_volume = self.lob_data[self.current_tick, 2]
        ask_volume = self.lob_data[self.current_tick, 3]
        return min(bid_volume, ask_volume)

# Example usage and training setup
def create_model_and_environment(lob_data: np.ndarray, config: Dict[str, Any]):
    """Create model and environment instances"""
    
    # Create environment
    env = LOBTradingEnvironment(
        lob_data=lob_data,
        lookback_window=config['lookback_window'],
        max_position_size=config['max_position_size'],
        transaction_cost=config['transaction_cost'],
        order_expiry_ticks=config['order_expiry_ticks'],
        initial_capital=config['initial_capital']
    )
    
    # Create model
    model = LSTMAttentionModel(
        input_dim=env.observation_space.shape[-1],
        hidden_dim=config['hidden_dim'],
        num_lstm_layers=config['num_lstm_layers'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout'],
        action_dim=env.action_space.n
    )
    
    return model, env

# Configuration example
config = {
    'lookback_window': 5,
    'max_position_size': 10.0,
    'transaction_cost': 0.001,
    'order_expiry_ticks': 10,
    'initial_capital': 100000.0,
    'hidden_dim': 128,
    'num_lstm_layers': 2,
    'num_attention_heads': 8,
    'dropout': 0.1
}