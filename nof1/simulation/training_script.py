import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import pickle
import os

# Assuming the previous model classes are imported
from lstm_attention_trading_model import LSTMAttentionModel, LOBTradingEnvironment, TradingAction, config, create_model_and_environment

class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with LSTM-Attention model for LOB trading"""
    
    def __init__(
        self,
        model: LSTMAttentionModel,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = model.to(device)
        self.target_network = LSTMAttentionModel(
            model.input_dim, model.hidden_dim, model.num_lstm_layers,
            model.attention1.num_heads, 0.0, len(TradingAction)
        ).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Hidden states for LSTM
        self.hidden_state = None
        self.target_hidden_state = None
    
    def get_epsilon(self) -> float:
        """Get current epsilon for epsilon-greedy exploration"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-1. * self.steps_done / self.epsilon_decay)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.get_epsilon():
            return random.randrange(len(TradingAction))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, _, self.hidden_state = self.q_network(state_tensor, self.hidden_state)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> Dict[str, float]:
        """Perform one step of learning"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = self.memory.Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q-values
        current_q_values, _, _ = self.q_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _, _ = self.target_network(next_state_batch)
            next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        return {
            'loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'q_values_mean': current_q_values.mean().item()
        }
    
    def reset_hidden_state(self):
        """Reset hidden states for new episode"""
        self.hidden_state = None
        self.target_hidden_state = None

class TradingTrainer:
    """Main training class for the trading agent"""
    
    def __init__(
        self,
        agent: DQNAgent,
        env: LOBTradingEnvironment,
        config: Dict[str, Any]
    ):
        self.agent = agent
        self.env = env
        self.config = config
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'total_trades': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }
    
    def train(self, num_episodes: int, save_freq: int = 100, log_freq: int = 10):
        """Train the agent"""
        
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            
            # Reset environment and agent
            state = self.env.reset()
            self.agent.reset_hidden_state()
            
            episode_reward = 0
            episode_length = 0
            portfolio_values = [self.env._get_portfolio_value()]
            
            while True:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Learn
                learning_metrics = self.agent.learn()
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                portfolio_values.append(info['portfolio_value'])
                
                if done:
                    break
            
            # Calculate episode metrics
            self._calculate_episode_metrics(episode, episode_reward, episode_length, 
                                          portfolio_values, info)
            
            # Logging
            if episode % log_freq == 0:
                self._log_metrics(episode, learning_metrics)
            
            # Save model
            if episode % save_freq == 0:
                self.save_model(f"model_episode_{episode}.pth")
        
        print("Training completed!")
        return self.training_metrics
    
    def _calculate_episode_metrics(self, episode: int, episode_reward: float, 
                                 episode_length: int, portfolio_values: List[float], 
                                 final_info: Dict):
        """Calculate and store episode metrics"""
        
        self.training_metrics['episode_rewards'].append(episode_reward)
        self.training_metrics['episode_lengths'].append(episode_length)
        self.training_metrics['portfolio_values'].append(portfolio_values[-1])
        self.training_metrics['total_trades'].append(final_info['total_trades'])
        
        # Calculate Sharpe ratio (simplified)
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            self.training_metrics['sharpe_ratios'].append(sharpe_ratio)
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
            self.training_metrics['max_drawdowns'].append(max_drawdown)
    
    def _log_metrics(self, episode: int, learning_metrics: Dict):
        """Log training metrics"""
        
        if len(self.training_metrics['episode_rewards']) == 0:
            return
        
        # Recent performance (last 10 episodes)
        recent_rewards = self.training_metrics['episode_rewards'][-10:]
        recent_portfolio_values = self.training_metrics['portfolio_values'][-10:]
        
        print(f"\nEpisode {episode}:")
        print(f"  Average Reward (last 10): {np.mean(recent_rewards):.4f}")
        print(f"  Average Portfolio Value (last 10): {np.mean(recent_portfolio_values):.2f}")
        print(f"  Epsilon: {self.agent.get_epsilon():.4f}")
        
        if learning_metrics:
            print(f"  Loss: {learning_metrics.get('loss', 0):.6f}")
            print(f"  Q-values Mean: {learning_metrics.get('q_values_mean', 0):.4f}")
        
        # Log to wandb if initialized
        if wandb.run:
            log_dict = {
                'episode': episode,
                'episode_reward': self.training_metrics['episode_rewards'][-1],
                'portfolio_value': self.training_metrics['portfolio_values'][-1],
                'epsilon': self.agent.get_epsilon(),
                'avg_reward_10': np.mean(recent_rewards),
                'avg_portfolio_value_10': np.mean(recent_portfolio_values)
            }
            
            if learning_metrics:
                log_dict.update(learning_metrics)
            
            if len(self.training_metrics['sharpe_ratios']) > 0:
                log_dict['sharpe_ratio'] = self.training_metrics['sharpe_ratios'][-1]
                log_dict['max_drawdown'] = self.training_metrics['max_drawdowns'][-1]
            
            wandb.log(log_dict)
    
    def save_model(self, filename: str):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        
        torch.save({
            'model_state_dict': self.agent.q_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': self.training_metrics,
            'steps_done': self.agent.steps_done
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        
        self.agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.steps_done = checkpoint['steps_done']
        self.training_metrics = checkpoint['training_metrics']
        
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained agent"""
        
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_metrics = {
            'episode_rewards': [],
            'portfolio_values': [],
            'total_trades': [],
            'sharpe_ratios': [],
            'win_rates': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            self.agent.reset_hidden_state()
            
            episode_reward = 0
            portfolio_values = [self.env._get_portfolio_value()]
            
            while True:
                # Select action (no exploration)
                action = self.agent.select_action(state, training=False)
                
                # Execute action
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                portfolio_values.append(info['portfolio_value'])
                
                if done:
                    break
            
            # Store metrics
            eval_metrics['episode_rewards'].append(episode_reward)
            eval_metrics['portfolio_values'].append(portfolio_values[-1])
            eval_metrics['total_trades'].append(info['total_trades'])
            
            # Calculate Sharpe ratio and win rate
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                win_rate = np.sum(returns > 0) / len(returns)
                
                eval_metrics['sharpe_ratios'].append(sharpe_ratio)
                eval_metrics['win_rates'].append(win_rate)
        
        # Calculate summary statistics
        summary = {
            'avg_episode_reward': np.mean(eval_metrics['episode_rewards']),
            'avg_portfolio_value': np.mean(eval_metrics['portfolio_values']),
            'avg_total_trades': np.mean(eval_metrics['total_trades']),
            'avg_sharpe_ratio': np.mean(eval_metrics['sharpe_ratios']),
            'avg_win_rate': np.mean(eval_metrics['win_rates'])
        }
        
        print("\nEvaluation Results:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")
        
        return summary

def prepare_sample_lob_data(num_ticks: int = 10000, num_features: int = 40) -> np.ndarray:
    """Generate sample LOB data for testing"""
    
    np.random.seed(42)
    
    # Generate realistic-looking LOB data
    base_price = 100.0
    price_volatility = 0.01
    
    lob_data = np.zeros((num_ticks, num_features))
    
    current_price = base_price
    
    for i in range(num_ticks):
        # Price movement (random walk with mean reversion)
        price_change = np.random.normal(0, price_volatility) - 0.001 * (current_price - base_price)
        current_price += price_change
        
        # Bid-ask spread
        spread = np.random.uniform(0.01, 0.05)
        bid_price = current_price - spread / 2
        ask_price = current_price + spread / 2
        
        # LOB levels (5 levels each side)
        for level in range(5):
            # Bid side
            lob_data[i, level * 2] = bid_price - level * 0.01  # Price
            lob_data[i, level * 2 + 1] = np.random.exponential(100)  # Volume
            
            # Ask side
            lob_data[i, (level + 5) * 2] = ask_price + level * 0.01  # Price
            lob_data[i, (level + 5) * 2 + 1] = np.random.exponential(100)  # Volume
        
        # Additional features (technical indicators, etc.)
        for j in range(20, num_features):
            lob_data[i, j] = np.random.normal(0, 1)
    
    return lob_data

# Main training script
def main():
    """Main training function"""
    
    # Configuration
    config = {
        # Environment parameters
        'lookback_window': 50,
        'max_position_size': 5.0,
        'transaction_cost': 0.001,
        'order_expiry_ticks': 5,
        'initial_capital': 100000.0,
        
        # Model parameters
        'hidden_dim': 128,
        'num_lstm_layers': 2,
        'num_attention_heads': 8,
        'dropout': 0.1,
        
        # Training parameters
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 5000,
        'buffer_size': 50000,
        'batch_size': 32,
        'target_update_freq': 500,
        
        # Training schedule
        'num_episodes': 1000,
        'save_freq': 100,
        'log_freq': 10
    }
    
    # Initialize wandb (optional)
    # wandb.init(project="lob-trading", config=config)
    
    # Prepare data
    print("Preparing LOB data...")
    lob_data = prepare_sample_lob_data(num_ticks=50000, num_features=40)
    
    # Create model and environment
    print("Creating model and environment...")
    model, env = create_model_and_environment(lob_data, config)
    
    # Create agent
    agent = DQNAgent(
        model=model,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq']
    )
    
    # Create trainer
    trainer = TradingTrainer(agent, env, config)
    
    # Train
    training_metrics = trainer.train(
        num_episodes=config['num_episodes'],
        save_freq=config['save_freq'],
        log_freq=config['log_freq']
    )
    
    # Evaluate
    eval_results = trainer.evaluate(num_episodes=50)
    
    # Save final model
    trainer.save_model("final_model.pth")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()