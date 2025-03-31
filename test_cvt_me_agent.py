#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the CVT-MAP-Elites trading agent with configuration from a YAML file.
"""

import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import argparse

# Import the CVTMAPElitesAgent class
# Assuming the agent is in the same directory
from src.agents.cvt_me_agent import CVTMAPElitesAgent
from src.simulation.env import TradingEnvironment
from src.utils.config_manager import ConfigManager
from src.data_ingestion.historical_data_reader import HistoricalDataReader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CVT-MAP-Elites trading agent')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'visualize'], 
                        default='train', help='Operation mode')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model (for test and visualize modes)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override number of episodes to run')
    return parser.parse_args()

def setup_logging(log_level):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_agent.log')
        ]
    )
    return logging.getLogger('trading_agent')

def create_directories(config):
    """Create necessary directories for saving models and plots."""
    # Create directory for saving the trained model
    save_path = config['training'].get('save_path', './trained_agents/model.pkl')
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create directory for plots if visualization is enabled
    if config.get('visualization', {}).get('save_plots', False):
        plot_dir = config['visualization'].get('plot_directory', './plots/')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def configure_agent(config, env):
    """Configure the agent from the loaded configuration."""
    # Extract behavior metrics and bounds
    behavior_metrics = config['behavior_space']['metrics']
    behavior_bounds = [
        config['behavior_space']['bounds'][metric] 
        for metric in behavior_metrics
    ]
    
    # Create agent configuration dictionary
    agent_config = {
        'name': config['name'],
        'behavior_dim': config['map_elites']['behavior_dim'],
        'num_centroids': config['map_elites']['num_centroids'],
        'genome_size': config['map_elites']['genome_size'],
        'eval_steps': config['training']['eval_steps'],
        'batch_size': config['training']['batch_size'],
        'num_init_samples': config['training']['num_init_samples'],
        'init_sigma': config['map_elites']['init_sigma'],
        'sigma_decay': config['map_elites']['sigma_decay'],
        'mutation_rate': config['map_elites']['mutation_rate'],
        'crossover_prob': config['map_elites']['crossover_prob'],
        'behavior_metrics': behavior_metrics,
        'behavior_bounds': behavior_bounds
    }
    
    # Create policy network configuration if specified
    if 'policy_network' in config:
        agent_config['hidden_layers'] = config['policy_network'].get('hidden_layers', [8])
        agent_config['activation'] = config['policy_network'].get('activation', 'tanh')
        agent_config['use_bias'] = config['policy_network'].get('use_bias', True)
    
    return CVTMAPElitesAgent(agent_config, env)

def train_agent(agent, config, logger):
    """Train the agent according to configuration."""
    # Get training parameters
    total_steps = config['training']['total_steps']
    save_path = config['training'].get('save_path', './trained_agents/model.pkl')
    checkpoint_freq = config['training'].get('checkpoint_frequency', 10)
    
    # Train the agent
    logger.info(f"Starting training for {total_steps} total steps")
    training_results = agent.train(total_steps)
    
    # Save the trained agent
    agent.save(save_path)
    logger.info(f"Agent saved to {save_path}")
    
    # Print training summary
    logger.info("Training summary:")
    for key, value in training_results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value}")
    
    return training_results

def test_agent(agent, env, episodes=10, logger=None):
    """Test the agent on the environment."""
    total_reward = 0
    returns = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        returns.append(info.get('total_pnl', 0.0))
        total_reward += episode_reward
        
        if logger:
            logger.info(f"Episode {episode+1}/{episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Return: {returns[-1]:.2f}")
    
    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return if std_return > 0 else 0
    
    if logger:
        logger.info(f"Test completed. Avg Return: {mean_return:.2f}, "
                  f"Std Dev: {std_return:.2f}, Sharpe: {sharpe:.2f}")
    
    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe_ratio": sharpe,
        "total_reward": total_reward,
        "returns": returns
    }

def visualize_archive(agent, config, logger=None):
    """Visualize the archive of discovered strategies."""
    viz_data = agent.get_archive_visualization_data()
    
    # Extract data
    behavior_descriptors = viz_data['behavior_descriptors']
    fitnesses = viz_data['fitnesses']
    metrics = viz_data['behavior_metrics']
    
    if len(behavior_descriptors) == 0:
        if logger:
            logger.warning("No strategies in archive to visualize")
        return
    
    # Create figure
    n_metrics = len(metrics)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each metric against fitness
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            metric_values = behavior_descriptors[:, i]
            
            # Scatter plot colored by fitness
            scatter = ax.scatter(metric_values, fitnesses, 
                               c=fitnesses, cmap='viridis', 
                               alpha=0.7, s=30)
            
            ax.set_xlabel(metric)
            ax.set_ylabel('Fitness')
            ax.set_title(f'{metric} vs Fitness')
            fig.colorbar(scatter, ax=ax)
    
    # Hide any unused axes
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save or show the figure
    if config.get('visualization', {}).get('save_plots', False):
        plot_dir = config['visualization'].get('plot_directory', './plots/')
        plot_path = os.path.join(plot_dir, 'archive_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"Archive visualization saved to {plot_path}")
    
    plt.show()
    
    # Also plot the training history
    if 'history' in viz_data and all(key in viz_data['history'] for key in ['mean_fitness', 'max_fitness', 'coverage']):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot fitness progression
        ax1.plot(viz_data['history']['mean_fitness'], label='Mean Fitness')
        ax1.plot(viz_data['history']['max_fitness'], label='Max Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Progression')
        ax1.legend()
        
        # Plot coverage
        ax2.plot(viz_data['history']['coverage'], label='Archive Coverage')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Coverage (%)')
        ax2.set_title('Archive Coverage')
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save or show the figure
        if config.get('visualization', {}).get('save_plots', False):
            plot_dir = config['visualization'].get('plot_directory', './plots/')
            plot_path = os.path.join(plot_dir, 'training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if logger:
                logger.info(f"Training history saved to {plot_path}")
        
        plt.show()

def print_elite_statistics(agent, logger=None):
    """Print statistics about the elite strategies in the archive."""
    if not agent.archive:
        if logger:
            logger.warning("No strategies in archive to analyze")
        return
    
    # Get statistics for each behavior metric
    metric_stats = agent.get_metric_statistics()
    
    if logger:
        logger.info("Archive statistics:")
        logger.info(f"  Total strategies: {len(agent.archive)}")
        logger.info(f"  Coverage: {len(agent.archive) / agent.num_centroids * 100:.1f}%")
        logger.info("  Metric statistics:")
        
        for metric, stats in metric_stats.items():
            logger.info(f"    {metric}: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                      f"mean={stats['mean']:.2f}, median={stats['median']:.2f}")
    
    # Find best strategies for each metric
    print("\nBest strategies by metric:")
    for i, metric in enumerate(agent.behavior_metrics):
        best_strategies = agent.get_best_individuals_by_metric(i, n=3)
        print(f"\n  Top 3 by {metric}:")
        for j, (individual, value) in enumerate(best_strategies):
            print(f"    {j+1}. {metric}: {value:.2f}, Fitness: {individual.fitness:.2f}")

def main():
    """Main entry point for running the agent."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config_manager = ConfigManager(args.config)
    
    # Set up logging
    log_level = config['training'].get('log_level', 'INFO')
    logger = setup_logging(log_level)
    
    # Create necessary directories
    create_directories(config)
    
    # Create environment
    data_reader = HistoricalDataReader(config_manager)
    data, _ = data_reader.preprocess_data()
    
    # Create environment
    env = TradingEnvironment(config_manager.config, data)
    
    # Operation mode
    if args.mode == 'train':
        # Configure and train agent
        agent = configure_agent(config, env)
        
        # Override episodes if specified
        if args.episodes is not None:
            config['training']['total_steps'] = args.episodes
            
        # Train the agent
        train_results = train_agent(agent, config, logger)
        
        # Visualize and analyze results
        visualize_archive(agent, config, logger)
        print_elite_statistics(agent, logger)
        
    elif args.mode in ['test', 'visualize']:
        # Load trained agent
        model_path = args.model or config['training'].get('save_path', './trained_agents/model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return
            
        # Configure agent
        agent = configure_agent(config, env)
        
        # Load saved state
        logger.info(f"Loading model from {model_path}")
        agent.load(model_path)
        
        if args.mode == 'test':
            # Test the agent
            episodes = args.episodes or 10
            test_results = test_agent(agent, env, episodes=episodes, logger=logger)
            
        elif args.mode == 'visualize':
            # Visualize the archive
            visualize_archive(agent, config, logger)
            print_elite_statistics(agent, logger)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()