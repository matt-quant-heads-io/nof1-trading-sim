import unittest
import numpy as np
from src.simulation.env import TradingEnvironment
from unittest.mock import MagicMock, patch

class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        # Create a minimal config for testing
        self.config = {
            'system': {'mode': 'historical'},
            'simulation': {
                'max_steps_per_episode': 100,
                'warmup_steps': 10,
                'initial_capital': 10000.0,
                'transaction_fee_pct': 0.001
            },
            'agents': {
                'positions': {'max_position': 10, 'min_position': -10},
                'observation_space': {'type': 'Box', 'low': -10, 'high': 10, 'shape': [10]},
                'action_space': {'type': 'Discrete', 'n': 3}
            },
            'data': {'historical': {'feature_columns': ['price', 'volume']}}
        }
        
        # Mock the OrderBook and RewardFunction
        with patch('src.simulation.env.OrderBook') as mock_orderbook_class:
            with patch('src.simulation.env.get_reward_function') as mock_reward_function:
                self.mock_orderbook = MagicMock()
                self.mock_reward = MagicMock()
                mock_orderbook_class.return_value = self.mock_orderbook
                mock_reward_function.return_value = self.mock_reward
                
                # Create the environment
                self.env = TradingEnvironment(self.config)
    
    def test_process_action_buy(self):
        """Test buying logic in _process_action"""
        # Setup
        self.env.position = 0.0
        self.env.capital = 10000.0
        self.env.realized_pnl = 0.0
        self.env.unrealized_pnl = 0.0
        self.env.last_price = 100.0
        
        # Mock the orderbook responses
        self.mock_orderbook.get_mid_price.return_value = 100.0
        self.mock_orderbook.calculate_execution_price.return_value = (100.0, 0.0)
        
        # Execute buy action (1)
        info = self.env._process_action(1)
        
        # Assertions
        self.assertEqual(self.env.position, 1.0)
        self.assertEqual(self.env.capital, 10000.0 - 100.0 - 0.1)  # price + fee
        self.assertEqual(self.env.transaction_costs, 0.1)  # 0.1% of 100
        self.assertEqual(info['position_change'], 1.0)
        self.assertEqual(info['execution_price'], 100.0)
    
    def test_process_action_sell(self):
        """Test selling logic in _process_action"""
        # Setup
        self.env.position = 1.0
        self.env.capital = 9899.9  # After buying at 100 with 0.1 fee
        self.env.realized_pnl = 0.0
        self.env.unrealized_pnl = 0.0
        self.env.last_price = 100.0
        
        # Mock the orderbook responses
        self.mock_orderbook.get_mid_price.return_value = 105.0
        self.mock_orderbook.calculate_execution_price.return_value = (105.0, 0.0)
        
        # Execute sell action (2)
        info = self.env._process_action(2)
        
        # Assertions
        self.assertEqual(self.env.position, 0.0)
        # Capital should increase by selling price minus fee
        self.assertEqual(round(self.env.capital, 1), round(9899.9 + 105.0 - 0.105, 1))
        self.assertEqual(round(self.env.realized_pnl, 1), 5.0)  # Profit from price difference
        self.assertEqual(round(self.env.transaction_costs, 3), 0.1 + 0.105)  # Both buy and sell fees
    
    def test_process_action_position_limits(self):
        """Test position limits in _process_action"""
        # Setup - already at max position
        self.env.position = 10.0  # Max position
        self.env.capital = 9000.0
        
        # Mock the orderbook responses
        self.mock_orderbook.get_mid_price.return_value = 100.0
        self.mock_orderbook.calculate_execution_price.return_value = (100.0, 0.0)
        
        # Try to buy more (should be limited)
        info = self.env._process_action(1)
        
        # Assertions - position should not change
        self.assertEqual(self.env.position, 10.0)
        self.assertEqual(info['position_change'], 0.0)

if __name__ == '__main__':
    unittest.main() 