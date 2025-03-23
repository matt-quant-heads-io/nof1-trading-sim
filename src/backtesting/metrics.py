import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(downside_returns)

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        equity_curve: Array of equity values
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate the running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage terms
    drawdown = (equity_curve - running_max) / running_max
    
    # Calculate the maximum drawdown
    max_drawdown = np.min(drawdown)
    
    return max_drawdown

def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate the win rate.
    
    Args:
        trades: DataFrame of trades with 'pnl' column
        
    Returns:
        Win rate as a percentage
    """
    if len(trades) == 0:
        return 0.0
    
    wins = len(trades[trades['pnl'] > 0])
    
    return wins / len(trades)

def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate the profit factor.
    
    Args:
        trades: DataFrame of trades with 'pnl' column
        
    Returns:
        Profit factor
    """
    if len(trades) == 0:
        return 0.0
    
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    total_profit = winning_trades['pnl'].sum()
    total_loss = abs(losing_trades['pnl'].sum())
    
    if total_loss == 0:
        return float('inf') if total_profit > 0 else 0.0
    
    return total_profit / total_loss

def calculate_average_trade(trades: pd.DataFrame) -> float:
    """
    Calculate the average trade P&L.
    
    Args:
        trades: DataFrame of trades with 'pnl' column
        
    Returns:
        Average trade P&L
    """
    if len(trades) == 0:
        return 0.0
    
    return trades['pnl'].mean()

def calculate_average_win(trades: pd.DataFrame) -> float:
    """
    Calculate the average winning trade P&L.
    
    Args:
        trades: DataFrame of trades with 'pnl' column
        
    Returns:
        Average winning trade P&L
    """
    winning_trades = trades[trades['pnl'] > 0]
    
    if len(winning_trades) == 0:
        return 0.0
    
    return winning_trades['pnl'].mean()

def calculate_average_loss(trades: pd.DataFrame) -> float:
    """
    Calculate the average losing trade P&L.
    
    Args:
        trades: DataFrame of trades with 'pnl' column
        
    Returns:
        Average losing trade P&L
    """
    losing_trades = trades[trades['pnl'] < 0]
    
    if len(losing_trades) == 0:
        return 0.0
    
    return losing_trades['pnl'].mean()

def calculate_expectancy(trades: pd.DataFrame) -> float:
    """
    Calculate the expectancy.
    
    Args:
        trades: DataFrame of trades with 'pnl' column
        
    Returns:
        Expectancy
    """
    win_rate = calculate_win_rate(trades)
    
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    if len(winning_trades) == 0 or len(losing_trades) == 0:
        return 0.0
    
    avg_win = winning_trades['pnl'].mean()
    avg_loss = abs(losing_trades['pnl'].mean())
    
    if avg_loss == 0:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    
    return (win_rate * win_loss_ratio) - (1 - win_rate)

def calculate_annualized_return(returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Calculate the annualized return.
    
    Args:
        returns: Array of returns
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate the compound return
    compound_return = np.prod(1 + returns) - 1
    
    # Annualize the return
    days = len(returns)
    annualized_return = (1 + compound_return) ** (trading_days_per_year / days) - 1
    
    return annualized_return

def calculate_annualized_volatility(returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility.
    
    Args:
        returns: Array of returns
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate the daily volatility
    daily_volatility = np.std(returns)
    
    # Annualize the volatility
    annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)
    
    return annualized_volatility

def calculate_calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Array of returns
        equity_curve: Array of equity values
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
    
    # Calculate the annualized return
    annualized_return = calculate_annualized_return(returns, trading_days_per_year)
    
    # Calculate the maximum drawdown
    max_drawdown = abs(calculate_max_drawdown(equity_curve))
    
    if max_drawdown == 0:
        return 0.0
    
    return annualized_return / max_drawdown

def calculate_metrics(equity_curve: np.ndarray, trades: pd.DataFrame, returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate all metrics.
    
    Args:
        equity_curve: Array of equity values
        trades: DataFrame of trades with 'pnl' column
        returns: Array of returns
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, equity_curve)
    
    # Return metrics
    metrics['total_return'] = (equity_curve[-1] / equity_curve[0]) - 1 if len(equity_curve) > 0 else 0.0
    metrics['annualized_return'] = calculate_annualized_return(returns)
    metrics['annualized_volatility'] = calculate_annualized_volatility(returns)
    
    # Trade metrics
    metrics['win_rate'] = calculate_win_rate(trades)
    metrics['profit_factor'] = calculate_profit_factor(trades)
    metrics['average_trade'] = calculate_average_trade(trades)
    metrics['average_win'] = calculate_average_win(trades)
    metrics['average_loss'] = calculate_average_loss(trades)
    metrics['expectancy'] = calculate_expectancy(trades)
    
    return metrics