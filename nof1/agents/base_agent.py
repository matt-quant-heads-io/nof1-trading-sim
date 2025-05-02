import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import gymnasium as gym
import torch
from nof1.simulation.env import TradingEnvironment

class BaseAgent:
    """
    Base class for trading agents.
    """
    def __init__(self, config: Dict[str, Any], env: gym.Env):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary
            env: Trading environment
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.env = env
        
        # Agent state
        self.name = config.get('name', 'BaseAgent')
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.current_observation = None
        
        # Initialize agent-specific statistics
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_returns = []
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Action to take
        """
        self.current_observation = observation
        
        # Base agent randomly selects an action
        return self.action_space.sample()
    
    def reset(self) -> None:
        """
        Reset the agent for a new episode.
        """
        self.episode_count += 1
        self.current_observation = None
    
    def train(self, num_episodes: int) -> Dict[str, Any]:
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Dictionary with training summary
        """
        self.logger.info(f"Training {self.name} for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            self.reset()
            
            terminated = False
            truncated = False
            episode_reward = 0
            
            while not (terminated or truncated):
                # Select action
                action = self.act(observation)
                
                # Take action
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Update agent
                self.update(observation, action, reward, next_observation, terminated or truncated, info)
                
                # Update state
                observation = next_observation
                episode_reward += reward
                self.total_steps += 1
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_returns.append(info.get('total_pnl', 0.0))
            
            if (episode + 1) % 10 == 0:
                self.logger.info(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Return: {info.get('total_pnl', 0.0):.2f}")
        
        # Return training summary
        return {
            "agent": self.name,
            "episodes": num_episodes,
            "total_steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards[-num_episodes:]),
            "mean_return": np.mean(self.episode_returns[-num_episodes:]),
            "std_reward": np.std(self.episode_rewards[-num_episodes:]),
            "std_return": np.std(self.episode_returns[-num_episodes:]),
            "max_reward": np.max(self.episode_rewards[-num_episodes:]),
            "min_reward": np.min(self.episode_rewards[-num_episodes:]),
            "max_return": np.max(self.episode_returns[-num_episodes:]),
            "min_return": np.min(self.episode_returns[-num_episodes:]),
        }
    
    def eval(self, config, test_states, test_prices, test_atrs, test_timestamps, num_episodes) -> Dict[str, Any]:
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Dictionary with training summary
        """
        self.logger.info(f"Training {self.name} for {num_episodes} episodes")
        self.eval_env = TradingEnvironment(config, states = test_states, prices=test_prices, atrs=test_atrs, timestamps=test_timestamps)
        
        with torch.no_grad():
            for episode in range(num_episodes):
                observation, info = self.eval_env.reset()
                self.reset()
                
                terminated = False
                truncated = False
                episode_reward = 0
                
                while not (terminated or truncated):
                    # Select action
                    action = self.act(observation)
                    
                    # Take action
                    next_observation, reward, terminated, truncated, info = self.eval_env.step(action)
                    
                    # Update agent
                    self.update(observation, action, reward, next_observation, terminated or truncated, info)
                    
                    # Update state
                    observation = next_observation
                    episode_reward += reward
                    self.total_steps += 1
                
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_returns.append(info.get('total_pnl', 0.0))
                
                if (episode + 1) % 10 == 0:
                    self.logger.info(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Return: {info.get('total_pnl', 0.0):.2f}")
        
        # Return training summary
        return {
            "agent": self.name,
            "episodes": num_episodes,
            "total_steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards[-num_episodes:]),
            "mean_return": np.mean(self.episode_returns[-num_episodes:]),
            "std_reward": np.std(self.episode_rewards[-num_episodes:]),
            "std_return": np.std(self.episode_returns[-num_episodes:]),
            "max_reward": np.max(self.episode_rewards[-num_episodes:]),
            "min_reward": np.min(self.episode_rewards[-num_episodes:]),
            "max_return": np.max(self.episode_returns[-num_episodes:]),
            "min_return": np.min(self.episode_returns[-num_episodes:]),
        }
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool, info: Dict[str, Any]) -> None:
        """
        Update the agent based on the environment interaction.
        
        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done
            info: Additional information
        """
        # Base agent does not learn
        pass
    
    def save(self, path: str) -> None:
        """
        Save the agent to the specified path.
        
        Args:
            path: Path to save the agent
        """
        # Base agent does not need to save anything
        pass
    
    def load(self, path: str) -> None:
        """
        Load the agent from the specified path.
        
        Args:
            path: Path to load the agent from
        """
        # Base agent does not need to load anything
        pass
