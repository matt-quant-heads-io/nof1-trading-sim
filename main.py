import os
import logging
import numpy as np
import pandas as pd
import argparse
import gymnasium as gym
from typing import Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

from nof1.utils.config_manager import ConfigManager
from nof1.data_ingestion.historical_data_reader import HistoricalDataReader
from nof1.data_ingestion.live_data_connector import LiveDataConnector
from nof1.simulation.env import TradingEnvironment
from nof1.agents.rl_agent import RLAgent
from nof1.backtesting.backtest_engine import BacktestEngine
from nof1.utils.benchmark import run_benchmark

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_levels.get(log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("trading_sim.log")
        ]
    )

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Trading Simulation System for RL Agents")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="config/experiment_config_2.yaml")
    parser.add_argument("--mode", type=str, choices=["historical", "paper_trading"], help="System mode")
    parser.add_argument("--data", type=str, help="Path to historical data file")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-steps", type=int, help="Number of steps for benchmark", default=10000)
    parser.add_argument("--episodes", type=int, help="Number of episodes", default=1)
    parser.add_argument("--timesteps", type=int, help="Number of timesteps for training", default=10000)
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    
    return parser.parse_args()


def plot_training_results(results, save_path):
    """
    Plot and save training results.
    
    Args:
        results: Dictionary containing training metrics
        save_path: Directory to save the plots
    """
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Plot rewards
    if 'episode_rewards' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(f"{save_path}/rewards_{timestamp}.png")
        plt.close()
    
    # Plot losses if available
    if 'losses' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['losses'])
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{save_path}/losses_{timestamp}.png")
        plt.close()
    
    # Plot any other metrics that might be in the results
    for key, values in results.items():
        if key not in ['episode_rewards', 'losses', 'algorithm'] and isinstance(values, list):
            # Check if the values are plottable (not dictionaries or other complex types)
            if values and all(not isinstance(item, dict) for item in values):
                plt.figure(figsize=(10, 6))
                plt.plot(values)
                plt.title(f'{key.replace("_", " ").title()}')
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.grid(True)
                plt.savefig(f"{save_path}/{key}_{timestamp}.png")
                plt.close()
    
    logging.info(f"Training plots saved to {save_path}")

def main():
    """
    Main function.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Override configuration with command line arguments
    if args.mode:
        config_manager.set("system.mode", args.mode)
    if args.data:
        config_manager.set("data.historical.data_path", args.data)
    if args.seed:
        config_manager.set("system.random_seed", args.seed)
    
    # Set up logging
    setup_logging(config_manager.get("system.log_level", "INFO"))
    
    # Set random seed
    np.random.seed(config_manager.get("system.random_seed", 42))
    
    # Get system mode
    mode = config_manager.get("system.mode")
    
    logging.info(f"Starting Trading Simulation System in {mode} mode")
    
    # Historical mode
    if mode == "historical":
        # Load and preprocess data
        data_reader = HistoricalDataReader(config_manager)
        states, prices, atrs, timestamps = data_reader.preprocess_data()
        
        # Create environment
        env = TradingEnvironment(config_manager.config, states = states, prices=prices, atrs=atrs, timestamps=timestamps)
        
        # Create agent
        agent = RLAgent(config_manager.config, env)
        
        # Load model if specified
        if args.model:
            try:
                agent.load(args.model)
                logging.info(f"Loaded model from {args.model}")
            except Exception as e:
                logging.error(f"Failed to load model from {args.model}: {e}")
        
        # Train the agent if specified
        if args.train:
            logging.info(f"Training agent for {args.timesteps} timesteps")
            
            # Create directories for models, logs, and results
            os.makedirs("models", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("results", exist_ok=True)
            
            # Train the agent
            train_results = agent.train(args.timesteps)
            
            # Plot and save training results
            plot_training_results(train_results, "./results")
            
            # Save the trained model
            model_path = f"models/agent_{mode}_{train_results['algorithm']}.zip"
            agent.save(model_path)
            logging.info(f"Saved trained model to {model_path}")
        
        # Run backtest if specified
        if args.backtest:
            logging.info(f"Running backtest for {args.episodes} episodes")
            
            # Create backtest engine
            backtest_engine = BacktestEngine(config_manager.config, env, agent)
            
            # Run backtest
            backtest_results = backtest_engine.run(args.episodes)
            
            # Log summary
            logging.info(f"Backtest results: Total Return: {backtest_results['total_return']:.2f}, Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}, Max Drawdown: {backtest_results.get('max_drawdown', 0):.2f}")
        
        # Run benchmark if specified
        if args.benchmark:
            benchmark_results = run_benchmark(args.config, args.benchmark_steps)
            print(f"Benchmark completed: {benchmark_results['steps_per_second']:.2f} steps/second")
            return
    
    # Paper trading mode
    elif mode == "paper_trading":
        # Create live data connector
        live_connector = LiveDataConnector(config_manager.config)
        
        # This is just a placeholder for paper trading mode
        # In a real implementation, you would create an environment that uses the live connector
        # and continuously feeds data to the agent
        
        logging.info("Paper trading mode is a placeholder in this implementation")
        logging.info("Starting mock paper trading session")
        
        # Connect to the exchange
        live_connector.connect()
        
        # Start the order book data feed
        try:
            live_connector.start_feed()
        except KeyboardInterrupt:
            logging.info("Paper trading session interrupted by user")
        finally:
            # Disconnect from the exchange
            live_connector.disconnect()
    
    else:
        logging.error(f"Unsupported mode: {mode}")

if __name__ == "__main__":
    main()