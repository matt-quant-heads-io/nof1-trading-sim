import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable

class RewardFunction:
    """
    Base class for reward functions.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward function.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
    
    def calculate_reward(self, 
                         action: int, 
                         position: float, 
                         position_change: float,
                         execution_price: float,
                         current_price: float,
                         unrealized_pnl: float,
                         realized_pnl: float,
                         transaction_cost: float,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the reward for the current step.
        
        Args:
            action: Agent action
            position: Current position
            position_change: Change in position due to action
            execution_price: Execution price for the trade
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            transaction_cost: Transaction cost
            info: Additional information
            
        Returns:
            Reward value
        """
        raise NotImplementedError("Subclasses must implement calculate_reward")


class SimplePnLReward(RewardFunction):
    """
    Simple reward function based on realized P&L and transaction costs.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
    def calculate_reward(self, 
                         action: int, 
                         position: float, 
                         position_change: float,
                         execution_price: float,
                         current_price: float,
                         unrealized_pnl: float,
                         realized_pnl: float,
                         transaction_cost: float,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the reward as the realized P&L minus transaction costs.
        
        Args:
            action: Agent action
            position: Current position
            position_change: Change in position due to action
            execution_price: Execution price for the trade
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            transaction_cost: Transaction cost
            info: Additional information
            
        Returns:
            Reward value
        """
        # import pdb; pdb.set_trace()
        if "nav_change" not in info:
            return 0.0
        reward = info["nav_change"]
        return reward


class RiskAdjustedReward(RewardFunction):
    """
    Risk-adjusted reward function that penalizes large positions and volatility.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.risk_aversion = config.get('risk_aversion', 0.1)
        self.position_penalty = config.get('position_penalty', 0.01)
        
    def calculate_reward(self, 
                         action: int, 
                         position: float, 
                         position_change: float,
                         execution_price: float,
                         current_price: float,
                         unrealized_pnl: float,
                         realized_pnl: float,
                         transaction_cost: float,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the risk-adjusted reward.
        
        Args:
            action: Agent action
            position: Current position
            position_change: Change in position due to action
            execution_price: Execution price for the trade
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            transaction_cost: Transaction cost
            info: Additional information
            
        Returns:
            Reward value
        """

        reward = info["nav_change"]
        
        return reward


class SharpeRatioReward(RewardFunction):
    """
    Reward function based on incremental Sharpe ratio.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.returns_history = []
        self.window_size = config.get('sharpe_window_size', 100)
        self.risk_free_rate = config.get('risk_free_rate', 0.0)
        
    def calculate_reward(self,
                         realized_pnl: float,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the reward based on incremental Sharpe ratio.
        
        Args:
            action: Agent action
            position: Current position
            position_change: Change in position due to action
            execution_price: Execution price for the trade
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            transaction_cost: Transaction cost
            info: Additional information
            
        Returns:
            Reward value
        """
        # Calculate return for this step
        step_return = realized_pnl
        
        # Add to history
        self.returns_history.append(step_return)
        
        # Keep only the last window_size returns
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
        
        # If we don't have enough history, use basic reward
        if len(self.returns_history) < 2:
            return step_return
        
        # Calculate Sharpe ratio
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns) - self.risk_free_rate
        std_return = np.std(returns)
        
        # Avoid division by zero
        if std_return == 0:
            return step_return
        
        sharpe = mean_return / std_return
        
        # Scale the reward based on Sharpe ratio
        # If Sharpe is negative, penalize more
        if sharpe < 0:
            reward = 2 * sharpe * abs(step_return)
        else:
            reward = sharpe * abs(step_return)
        
        return reward


class SortinoRatioReward(RewardFunction):
    """
    Reward function based on incremental Sortino ratio.
    Sortino ratio is similar to Sharpe ratio but only penalizes downside volatility.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.returns_history = []
        self.window_size = config.get('sortino_window_size', 10)
        self.risk_free_rate = config.get('risk_free_rate', 0.03)
        self.target_return = config.get('target_return', 0.0)
        
    def calculate_reward(self,
                         realized_pnl: float,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the reward based on incremental Sortino ratio.
        
        Args:
            action: Agent action
            position: Current position
            position_change: Change in position due to action
            execution_price: Execution price for the trade
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            transaction_cost: Transaction cost
            info: Additional information
            
        Returns:
            Reward value
        """
        # Calculate return for this step
        step_return = realized_pnl
        
        # Add to history
        self.returns_history.append(step_return)
        
        # Keep only the last window_size returns
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
        
        # If we don't have enough history, use basic reward
        if len(self.returns_history) < 2:
            return step_return
        
        # Calculate Sortino ratio
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns) - self.risk_free_rate
        
        # Calculate downside deviation (only negative returns below target)
        downside_returns = returns[returns < self.target_return] - self.target_return
        
        # If no downside returns, use a small value to avoid division by zero
        if len(downside_returns) == 0:
            downside_deviation = 1e-6
        else:
            downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
        
        # Avoid division by zero
        if downside_deviation == 0:
            downside_deviation = 1e-6
        
        sortino = mean_return / downside_deviation
        
        # Scale the reward based on Sortino ratio
        # If Sortino is negative, penalize more
        if sortino < 0:
            reward = 2 * sortino * abs(step_return)
        else:
            reward = sortino * abs(step_return)
        
        return reward


class CalmarRatioReward(RewardFunction):
    """
    Reward function based on incremental Calmar ratio.
    Calmar ratio is the ratio of annualized return over the maximum drawdown.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.returns_history = []
        self.cumulative_returns = [1.0]  # Start with initial value of 1.0
        self.window_size = config.get('calmar_window_size', 5)  # Default to ~1 year of trading days
        
    def calculate_reward(self,
                         realized_pnl: float,
                         info: Dict[str, Any]) -> float:
        """
        Calculate the reward based on incremental Calmar ratio.
        
        Args:
            action: Agent action
            position: Current position
            position_change: Change in position due to action
            execution_price: Execution price for the trade
            current_price: Current market price
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            transaction_cost: Transaction cost
            info: Additional information
            
        Returns:
            Reward value
        """
        # Calculate return for this step
        step_return = realized_pnl
        
        # Add to history
        self.returns_history.append(step_return)
        
        # Update cumulative returns
        self.cumulative_returns.append(self.cumulative_returns[-1] * (1 + step_return))
        
        # Keep only the last window_size returns and cumulative returns
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
            self.cumulative_returns.pop(0)
        
        # If we don't have enough history, use basic reward
        if len(self.returns_history) < 10:  # Need some history to calculate meaningful drawdown
            return step_return
        
        # Calculate Calmar ratio components
        cumulative_returns = np.array(self.cumulative_returns)
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Calculate annualized return (approximation)
        total_return = cumulative_returns[-1] / cumulative_returns[0] - 1
        period_fraction = len(self.returns_history) / 252  # Assuming 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / period_fraction) - 1 if period_fraction > 0 else total_return
        
        # Avoid division by zero
        if max_drawdown == 0:
            max_drawdown = 1e-6
        
        calmar = annualized_return / max_drawdown
        
        # Scale the reward based on Calmar ratio
        # If Calmar is negative, penalize more
        if calmar < 0:
            reward = 2 * calmar * abs(step_return)
        else:
            reward = calmar * abs(step_return)
        
        return reward


def get_reward_function(config: Dict[str, Any]) -> RewardFunction:
    """
    Factory function to get the specified reward function.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Reward function instance
    """
    reward_type = config['rl']['reward_type']
    
    if reward_type == 'simple_pnl':
        return SimplePnLReward(config)
    elif reward_type == 'risk_adjusted':
        return RiskAdjustedReward(config)
    elif reward_type == 'sharpe_ratio':
        return SharpeRatioReward(config)
    elif reward_type == 'sortino_ratio':
        return SortinoRatioReward(config)
    elif reward_type == 'calmar_ratio':
        return CalmarRatioReward(config)
    else:
        logging.warning(f"Unknown reward type '{reward_type}', using simple P&L reward")
        return SimplePnLReward(config)