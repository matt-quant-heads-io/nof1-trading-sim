import os
import numpy as np
import logging
import gymnasium as gym
from typing import Dict, Any, List, Tuple, Optional, Union, Type
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.callbacks import BaseCallback

from nof1.agents.base_agent import BaseAgent

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging to tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_returns = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        """
        Called after each step of the environment during training.
        """
        # Get the action mask from the environment info
        action_mask = self.locals['infos'][0].get('action_mask', None)
        
        # Store it for use in the next action selection
        self.action_mask = action_mask
        
        # Update episode stats
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Check if episode is done (terminated or truncated)
        if self.locals["dones"][0] or self.locals.get("truncated", [False])[0]:
            # Get episode info
            info = self.locals["infos"][0]
            episode_return = info.get("total_pnl", 0.0)
            
            # Store episode stats
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_returns.append(episode_return)
            
            # Log to tensorboard
            self.logger.record("episode_reward", self.current_episode_reward)
            self.logger.record("episode_length", self.current_episode_length)
            self.logger.record("episode_return", episode_return)
            
            # Reset episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


class RLAgent(BaseAgent):
    """
    Reinforcement learning agent using Stable Baselines3.
    """
    def __init__(self, config: Dict[str, Any], env: gym.Env):
        """
        Initialize the RL agent.
        
        Args:
            config: Configuration dictionary
            env: Trading environment
        """
        super(RLAgent, self).__init__(config, env)
        self.config = config
        self.config.rl
        # RL configuration
        self.algorithm = self.config.rl.algorithm
        self.policy_type = self.config.rl.policy_type 
        self.learning_rate = self.config.rl.learning_rate 
        self.tensorboard_log = self.config.rl.tensorboard_log 
        self.device = self.config.rl.device
        
        # Create model
        self.model = self._create_model()
    
    def _create_model(self):
        """
        Create the RL model based on configuration.
        
        Returns:
            Stable Baselines3 model
        """
        # Extract algorithm-specific parameters
        algo_params = {}
        
        # Common parameters
        algo_params["learning_rate"] = self.learning_rate
        algo_params["tensorboard_log"] = self.tensorboard_log
        algo_params["verbose"] = self.config.rl.verbose
        algo_params["device"] = self.device
        
        # Algorithm-specific parameters
        if self.algorithm == 'PPO':
            algo_params["n_steps"] = self.config.rl.n_steps
            algo_params["batch_size"] = self.config.rl.batch_size
            algo_params["n_epochs"] = self.config.rl.n_epochs
            algo_params["gamma"] = self.config.rl.gamma
            algo_params["gae_lambda"] = self.config.rl.gae_lambda
            algo_params["clip_range"] = self.config.rl.clip_range
            algo_params["normalize_advantage"] = self.config.rl.normalize_advantage
            algo_params["ent_coef"] = self.config.rl.ent_coef
            algo_params["vf_coef"] = self.config.rl.vf_coef
            algo_params["max_grad_norm"] = self.config.rl.max_grad_norm
            algo_params["use_sde"] = self.config.rl.use_sde
            algo_params["sde_sample_freq"] = self.config.rl.sde_sample_freq
            algo_params["target_kl"] = self.config.rl.target_kl
            
            return PPO(self.policy_type, self.env, **algo_params)
            
        elif self.algorithm == 'A2C':
            algo_params["n_steps"] = self.config.get('rl.n_steps', 5)
            algo_params["gamma"] = self.config.get('rl.gamma', 0.99)
            algo_params["gae_lambda"] = self.config.get('rl.gae_lambda', 1.0)
            algo_params["ent_coef"] = self.config.get('rl.ent_coef', 0.0)
            algo_params["vf_coef"] = self.config.get('rl.vf_coef', 0.5)
            algo_params["max_grad_norm"] = self.config.get('rl.max_grad_norm', 0.5)
            algo_params["use_rms_prop"] = self.config.get('rl.use_rms_prop', True)
            algo_params["normalize_advantage"] = self.config.get('rl.normalize_advantage', False)
            
            return A2C(self.policy_type, self.env, **algo_params)
            
        elif self.algorithm == 'DQN':
            algo_params["buffer_size"] = self.config.get('rl.buffer_size', 1000000)
            algo_params["learning_starts"] = self.config.get('rl.learning_starts', 50000)
            algo_params["batch_size"] = self.config.get('rl.batch_size', 32)
            algo_params["tau"] = self.config.get('rl.tau', 1.0)
            algo_params["gamma"] = self.config.get('rl.gamma', 0.99)
            algo_params["train_freq"] = self.config.get('rl.train_freq', 4)
            algo_params["gradient_steps"] = self.config.get('rl.gradient_steps', 1)
            algo_params["target_update_interval"] = self.config.get('rl.target_update_interval', 10000)
            algo_params["exploration_fraction"] = self.config.get('rl.exploration_fraction', 0.1)
            algo_params["exploration_initial_eps"] = self.config.get('rl.exploration_initial_eps', 1.0)
            algo_params["exploration_final_eps"] = self.config.get('rl.exploration_final_eps', 0.05)
            algo_params["max_grad_norm"] = self.config.get('rl.max_grad_norm', 10)
            
            return DQN(self.policy_type, self.env, **algo_params)
            
        elif self.algorithm == 'SAC':
            algo_params["buffer_size"] = self.config.get('rl.buffer_size', 1000000)
            algo_params["learning_starts"] = self.config.get('rl.learning_starts', 100)
            algo_params["batch_size"] = self.config.get('rl.batch_size', 256)
            algo_params["tau"] = self.config.get('rl.tau', 0.005)
            algo_params["gamma"] = self.config.get('rl.gamma', 0.99)
            algo_params["train_freq"] = self.config.get('rl.train_freq', 1)
            algo_params["gradient_steps"] = self.config.get('rl.gradient_steps', 1)
            algo_params["action_noise"] = None  # Could add noise here
            algo_params["ent_coef"] = self.config.get('rl.ent_coef', 'auto')
            algo_params["target_update_interval"] = self.config.get('rl.target_update_interval', 1)
            algo_params["target_entropy"] = self.config.get('rl.target_entropy', 'auto')
            algo_params["use_sde"] = self.config.get('rl.use_sde', False)
            algo_params["sde_sample_freq"] = self.config.get('rl.sde_sample_freq', -1)
            
            return SAC(self.policy_type, self.env, **algo_params)
            
        else:
            self.logger.error(f"Unsupported algorithm: {self.algorithm}")
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def act(self, observation: np.ndarray, action_mask: np.ndarray = None) -> int:
        """
        Select an action based on the observation and action mask.
        
        Args:
            observation: Environment observation
            action_mask: Binary mask indicating valid actions (1=valid, 0=invalid)
            
        Returns:
            Action to take
        """
        self.current_observation = observation
        
        # Use the trained model to predict action
        action_values = self.model.predict(observation, deterministic=False)[1]
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid action probabilities to -inf (or a very low value)
            masked_action_values = action_values.copy()
            masked_action_values[action_mask == 0] = float('-inf')
            
            # Select the action with the highest valid value
            action = np.argmax(masked_action_values)
        else:
            # If no mask provided, use the model's prediction directly
            action, _ = self.model.predict(observation, deterministic=True)
        
        return int(action)
    
    def train(self, num_timesteps: int) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            num_timesteps: Number of timesteps to train
            
        Returns:
            Dictionary with training summary
        """
        self.logger.info(f"Training {self.name} ({self.algorithm}) for {num_timesteps} timesteps")
        
        # Create tensorboard callback
        callback = TensorboardCallback()
        
        # Train the model
        self.model.learn(total_timesteps=num_timesteps, callback=callback)
        
        # Return training summary
        return {
            "agent": self.name,
            "algorithm": self.algorithm,
            "timesteps": num_timesteps,
            "mean_reward": np.mean(callback.episode_rewards).item() if callback.episode_rewards else 0,
            "mean_return": np.mean(callback.episode_returns) if callback.episode_returns else 0,
            "episode_rewards": [float(r) for r in callback.episode_rewards],
            "episode_returns": [float(r) for r in callback.episode_returns],
            "mean_episode_length": np.mean(callback.episode_lengths).item() if callback.episode_lengths else 0,
        }
    
    def save(self, path: str) -> None:
        """
        Save the RL model.
        
        Args:
            path: Path to save the model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        self.logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the RL model.
        
        Args:
            path: Path to load the model from
        """
        # Map algorithm names to classes
        algo_map = {
            "PPO": PPO,
            "A2C": A2C,
            "DQN": DQN,
            "SAC": SAC,
            "TD3": TD3
        }
        
        # Load the model
        if self.algorithm in algo_map:
            self.model = algo_map[self.algorithm].load(path, env=self.env)
            self.logger.info(f"Loaded model from {path}")
        else:
            self.logger.error(f"Unsupported algorithm for loading: {self.algorithm}")
            raise ValueError(f"Unsupported algorithm for loading: {self.algorithm}")
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool, info: Dict[str, Any]) -> None:
        """
        Update method is not needed for SB3 agents as the training is handled by the learn method.
        """
        pass
