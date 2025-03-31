import numpy as np
from src.simulation.env import TradingEnvironment
from src.simulation.orderbook import OrderBook
from unittest.mock import MagicMock, patch
from src.utils.config_manager import ConfigManager

def test_trading_metrics():
    # Create a minimal config
    config = {
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
    ConfigManager(config)
    
    # Create a mock OrderBook
    with patch('src.simulation.env.OrderBook') as mock_orderbook_class:
        with patch('src.simulation.env.get_reward_function'):
            mock_orderbook = MagicMock()
            mock_orderbook_class.return_value = mock_orderbook
            
            # Create the environment
            env = TradingEnvironment(config)
            
            # Test scenario 1: Buy and then sell with profit
            print("=== Scenario 1: Buy and Sell with Profit ===")
            env.position = 0.0
            env.capital = 10000.0
            env.realized_pnl = 0.0
            env.unrealized_pnl = 0.0
            env.last_price = 100.0
            
            # Mock the orderbook responses for buy
            mock_orderbook.get_mid_price.return_value = 100.0
            mock_orderbook.calculate_execution_price.return_value = (100.0, 0.0)
            
            # Execute buy action
            buy_info = env._process_action(1)
            print(f"After Buy: Position={env.position}, Capital={env.capital}, Realized PnL={env.realized_pnl}")
            
            # Mock the orderbook responses for sell (price increased)
            mock_orderbook.get_mid_price.return_value = 105.0
            mock_orderbook.calculate_execution_price.return_value = (105.0, 0.0)
            
            # Execute sell action
            sell_info = env._process_action(2)
            print(f"After Sell: Position={env.position}, Capital={env.capital}, Realized PnL={env.realized_pnl}")
            
            # Test scenario 2: Buy and then sell with loss
            print("\n=== Scenario 2: Buy and Sell with Loss ===")
            env.position = 0.0
            env.capital = 10000.0
            env.realized_pnl = 0.0
            env.unrealized_pnl = 0.0
            env.last_price = 100.0
            
            # Mock the orderbook responses for buy
            mock_orderbook.get_mid_price.return_value = 100.0
            mock_orderbook.calculate_execution_price.return_value = (100.0, 0.0)
            
            # Execute buy action
            buy_info = env._process_action(1)
            print(f"After Buy: Position={env.position}, Capital={env.capital}, Realized PnL={env.realized_pnl}")
            
            # Mock the orderbook responses for sell (price decreased)
            mock_orderbook.get_mid_price.return_value = 95.0
            mock_orderbook.calculate_execution_price.return_value = (95.0, 0.0)
            
            # Execute sell action
            sell_info = env._process_action(2)
            print(f"After Sell: Position={env.position}, Capital={env.capital}, Realized PnL={env.realized_pnl}")
            
            # Test scenario 3: Position limits
            print("\n=== Scenario 3: Position Limits ===")
            env.position = 10.0  # Max position
            env.capital = 9000.0
            
            # Try to buy more (should be limited)
            mock_orderbook.get_mid_price.return_value = 100.0
            mock_orderbook.calculate_execution_price.return_value = (100.0, 0.0)
            
            limit_info = env._process_action(1)
            print(f"After Buy at Max Position: Position={env.position}, Position Change={limit_info['position_change']}")

if __name__ == "__main__":
    test_trading_metrics() 