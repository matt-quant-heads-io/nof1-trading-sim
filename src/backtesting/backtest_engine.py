import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import time
import json
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.simulation.env import TradingEnvironment
from src.backtesting.metrics import calculate_metrics

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    def __init__(self, config: Dict[str, Any], env: TradingEnvironment, agent: BaseAgent):
        """
        Initialize the backtest engine.
        
        Args:
            config: Configuration dictionary
            env: Trading environment
            agent: Trading agent
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.env = env
        self.agent = agent
        
        # Backtesting configuration
        self.metrics_list = self._get_config_value('backtesting.metrics', [])
        self.results_dir = self._get_config_value('backtesting.results_dir', './results')
        self.plot_results = self._get_config_value('backtesting.plot_results', True)
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.trades = pd.DataFrame()
        self.equity_curve = []
        self.returns = []
        self.positions = []
        self.timestamps = []
        
    def _get_config_value(self, path: str, default: Any = None) -> Any:
        """
        Get a value from the config using dot notation.
        
        Args:
            path: Path to the config value using dot notation (e.g., 'backtesting.metrics')
            default: Default value to return if the path doesn't exist
            
        Returns:
            The config value or the default
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def run(self, num_episodes: int = 1) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Running backtest for {num_episodes} episodes")
        
        start_time = time.time()
        
        # Clear previous results
        self.equity_curve = []
        self.returns = []
        self.positions = []
        self.timestamps = []
        self.trades = pd.DataFrame()
        
        all_trades = []
        
        for episode in range(num_episodes):
            # Reset the environment
            observation, info = self.env.reset()
            
            # Initial equity
            initial_equity = self.env.capital
            episode_equity = [initial_equity]
            episode_returns = [0.0]
            episode_positions = [0.0]
            episode_timestamps = [0]
            
            terminated = False
            truncated = False
            step = 0
            
            while not (terminated or truncated):
                # Get action from agent
                action = self.agent.act(observation)
                
                # Take a step in the environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Record equity, returns, and positions
                current_equity = info['capital'] + info['position'] * info.get('price', 0)
                episode_equity.append(current_equity)
                
                # Calculate return for this step
                if len(episode_equity) > 1 and episode_equity[-2] > 0:
                    step_return = episode_equity[-1] / episode_equity[-2] - 1
                else:
                    step_return = 0.0
                
                episode_returns.append(step_return)
                episode_positions.append(info['position'])
                episode_timestamps.append(step + 1)
                
                # Record trades
                if info.get('position_change', 0) != 0:
                    trade = {
                        'episode': episode,
                        'step': step,
                        'timestamp': step + 1,
                        'type': 'buy' if info.get('position_change', 0) > 0 else 'sell',
                        'price': info.get('execution_price', info.get('price', 0)),
                        'size': abs(info.get('position_change', 0)),
                        'cost': abs(info.get('position_change', 0) * info.get('execution_price', info.get('price', 0))),
                        'fee': info.get('transaction_cost', 0),
                        'pnl': info.get('realized_pnl_after', 0) - info.get('realized_pnl_before', 0),
                    }
                    all_trades.append(trade)
                
                # Update observation
                observation = next_observation
                step += 1
            
            # Log episode results
            final_equity = episode_equity[-1]
            total_return = (final_equity / initial_equity) - 1
            
            self.logger.info(f"Episode {episode + 1}/{num_episodes}, Return: {total_return:.2f}, Steps: {step}")
            
            # Append episode results to overall results
            self.equity_curve.extend(episode_equity)
            self.returns.extend(episode_returns)
            self.positions.extend(episode_positions)
            self.timestamps.extend(episode_timestamps)
        
        # Process trades
        if all_trades:
            self.trades = pd.DataFrame(all_trades)
        
        # Calculate metrics
        self.results = self._calculate_metrics()
        
        # Add basic statistics
        self.results.update({
            'num_episodes': num_episodes,
            'total_steps': len(self.equity_curve) - 1,
            'total_trades': len(self.trades),
            'backtest_duration': time.time() - start_time,
            'final_equity': self.equity_curve[-1] if self.equity_curve else 0,
            'initial_equity': self.equity_curve[0] if self.equity_curve else 0,
            'total_return': (self.equity_curve[-1] / self.equity_curve[0] - 1) if len(self.equity_curve) > 1 else 0,
        })
        
        # Save results
        self._save_results()
        
        # Plot results
        if self.plot_results:
            self._plot_results()
        
        return self.results
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics for the backtest.
        
        Returns:
            Dictionary with metrics
        """
        if not self.equity_curve or len(self.equity_curve) <= 1:
            self.logger.warning("Insufficient data for metric calculation")
            return {}
        
        # Prepare data for metrics calculation
        equity_curve = np.array(self.equity_curve[1:])  # Skip initial equity
        returns = np.array(self.returns[1:])  # Skip initial return (0)
        
        # Calculate metrics if trades exist
        if not self.trades.empty:
            metrics = calculate_metrics(equity_curve, self.trades, returns)
        else:
            # Basic metrics if no trades
            metrics = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
            }
        
        return metrics
    
    def _save_results(self) -> None:
        """
        Save backtest results to files.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"backtest_{timestamp}"
        
        # Save results dictionary
        results_file = os.path.join(self.results_dir, f"{base_filename}_results.json")
        with open(results_file, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in self.results.items()}, f, indent=4)
        
        # Save equity curve
        equity_file = os.path.join(self.results_dir, f"{base_filename}_equity.csv")
        equity_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve,
            'return': self.returns,
            'position': self.positions,
        })
        equity_df.to_csv(equity_file, index=False)
        
        # Save trades
        if not self.trades.empty:
            trades_file = os.path.join(self.results_dir, f"{base_filename}_trades.csv")
            self.trades.to_csv(trades_file, index=False)
        
        self.logger.info(f"Saved backtest results to {self.results_dir}")
    
    def _plot_results(self) -> None:
        """
        Plot backtest results.
        """
        if not self.equity_curve or len(self.equity_curve) <= 1:
            self.logger.warning("Insufficient data for plotting")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.results_dir, f"backtest_{timestamp}_plot.png")
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot equity curve
        ax1.plot(self.timestamps, self.equity_curve)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Plot returns
        ax2.plot(self.timestamps, self.returns)
        ax2.set_title('Returns')
        ax2.set_ylabel('Return')
        ax2.grid(True)
        
        # Plot positions
        ax3.plot(self.timestamps, self.positions)
        ax3.set_title('Positions')
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Step')
        ax3.grid(True)
        
        # Add metrics as text
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in self.results.items() if isinstance(v, (int, float, np.number))])
        fig.text(0.02, 0.02, metrics_text, verticalalignment='bottom', horizontalalignment='left', fontsize=10)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close(fig)
        
        self.logger.info(f"Saved backtest plot to {plot_file}")