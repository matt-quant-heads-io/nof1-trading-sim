import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

from src.simulation.orderbook import OrderBook
from src.simulation.rewards import get_reward_function, RewardFunction

class TradingEnvironment(gym.Env):
    """
    Trading environment for RL agents.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict[str, Any], data: Optional[np.ndarray] = None):
        # Initialize the Gymnasium environment
        super(TradingEnvironment, self).__init__()
        """
        Initialize the trading environment.
        
        Args:
            config: Configuration dictionary
            data: Historical data as numpy array (if in historical mode)
        """
        super(TradingEnvironment, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # System configuration
        self.mode = self.config.system.mode
        self.execution_price = None
        self.capital = None
        # Environment configuration
        self.max_steps = self.config.simulation.max_steps_per_episode
        self.warmup_steps = self.config.simulation.warmup_steps

        self.initial_capital = self.config.simulation.initial_capital
        self.transaction_fee_pct = self.config.simulation.transaction_fee_pct
        self.position_size_fixed_dollar = self.config.simulation.position_size_fixed_dollar
        
        # Agent configuration
        self.max_position = self.config.agents.positions.max_position
        self.min_position = self.config.agents.positions.min_position
        
        # Setup feature columns
        self.feature_columns = self.config.data.historical.feature_columns
        
        # Setup spaces
        obs_space_type = self.config.agents.observation_space.type 
        action_space_type = self.config.agents.action_space.type
        self._infos = []
        
        if obs_space_type == 'Box':
            obs_low = self.config.agents.observation_space.low
            obs_high = self.config.agents.observation_space.high
            obs_shape = (self.config.agents.observation_space.n_feats * self.config.agents.observation_space.n_stack,)
            
            self.observation_space = spaces.Box(
                low=float(obs_low),
                high=float(obs_high),
                shape=tuple(obs_shape),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported observation space type: {obs_space_type}")
        
        if action_space_type == 'Discrete':
            action_n = self.config.agents.action_space.n # 0: hold, 1: buy, 2: sell
            self.action_space = spaces.Discrete(action_n)
        else:
            raise ValueError(f"Unsupported action space type: {action_space_type}")
        
        # Initialize orderbook
        self.orderbook = OrderBook(config)
        
        # Initialize reward function
        self.reward_function = get_reward_function(config)
        
        # Data for historical mode
        self.data = data
        self.current_step = 0
        self.episode_step = 0
        
        # Add this line to store the current state
        self.current_state = {}
        
        # Trading state
        self.position = 0.0
        self.capital = self.initial_capital
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_price = None
        self.transaction_costs = 0.0
        self._cum_nav_chg = 0.0
        self.long_trade_wins = 0
        self.short_trade_wins = 0
        self.long_trades = 0
        self.short_trades = 0
        self.overall_win_pct = 0.0
        self.short_win_pct = 0.0
        self.long_win_pct = 0.0
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_positions = []
        self.episode_trades = []
        self.episode_pnls = []
        self.stacked_obs = None
        
        # Initialize state
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for random number generator
            options: Additional options
            
        Returns:
            Tuple of (initial observation, info dict)
        """
        # Set the seed if provided
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        
        # Reset trading state
        self.position = 0.0
        self.capital = self.initial_capital
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.transaction_costs = 0.0
        
        # Reset episode counters
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_positions = []
        self.episode_trades = []
        self.episode_pnls = []
        self.stacked_obs = None
        
        if self.mode == 'historical' and self.data is not None:
            # Start at a random point for each episode, allowing for max_steps
            max_start = max(0, len(self.data) - self.max_steps - 1)
            self.current_step = np.random.randint(0, max_start) if max_start > 0 else 0
            
            # Apply warmup steps to initialize orderbook state
            for i in range(self.warmup_steps):
                if self.current_step + i < len(self.data):
                    self._update_orderbook_historical(self.current_step + i)
            
            # Set current step to after warmup
            self.current_step += self.warmup_steps
            
            # Update orderbook with the current step data
            if self.current_step < len(self.data):
                self._update_orderbook_historical(self.current_step)
        
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial info
        info = {
            "position": self.position,
            "capital": self.capital,
            "step": self.episode_step,
            "action_mask": self._get_action_mask()
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Agent action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.mode == 'historical' and self.data is not None:
            # Check if we're at the end of the data
            if self.current_step >= len(self.data) - 1:
                return self._get_observation(), 0.0, True, False, {"reason": "end_of_data"}
        
        # Process the action
        info = self._process_action(action)
        
        # Update environment state
        if self.mode == 'historical' and self.data is not None:
            self.current_step += 1
            if self.current_step < len(self.data):
                self._update_orderbook_historical(self.current_step)
        
        # Update episode step counter
        self.episode_step += 1
        
        # Get new observation
        observation = self._get_observation()
        # print(f"observation: {observation}")
        
        # Calculate reward
        reward = self._calculate_reward(action, info)
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        terminated = self._is_episode_done()
        # Truncated is used when episode is cut off prematurely (e.g., due to time limits)
        truncated = False

        self._infos.append(info)
        
        if terminated:
            # Add episode summary to info
            info.update(self._get_episode_summary())

        return observation, reward, terminated, truncated, info
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Get a binary mask indicating which actions are valid in the current state.
        
        Returns:
            Binary mask where 1 = valid action, 0 = invalid action
        """
        # Initialize mask with all actions valid
        mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # If at max position, can't buy more
        if self.position >= self.max_position:
            mask[1] = 0  # Disable buy action
        
        # If at min position, can't sell more
        if self.position <= self.min_position:
            mask[2] = 0  # Disable sell action
            
        
        return mask

    def _process_onehot_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Process the agent's (one-hot encoded) action.

        Args:
            action: Agent action
            
        Returns:
            Dictionary with action information
        """
        disc_action = np.argmax(action)
        return self._process_action(disc_action)
    
    def _process_action(self, action: int) -> Dict[str, Any]:
        """
        Process the agent's action.
        
        Args:
            action: Agent action
            
        Returns:
            Dictionary with action information
        """
        action = int(action)
        
        # Get action mask and validate action
        action_mask = self._get_action_mask()
        if action_mask[action] == 0:
            # self.logger.warning(f"Invalid action {action} selected with position {self.position}. Converting to no-op.")
            action = 0  # Convert to no-op (hold)
        
        # Get current price
        current_price = self.orderbook.get_mid_price()
        if current_price is None:
            self.logger.warning("No mid price available, using last price")
            current_price = self.last_price
        
        if current_price is None:
            self.logger.error("No price information available")
            return {"success": False, "reason": "no_price"}
        
        # Store the last price
        self.last_price = current_price
        prior_nav = self.capital
        self.max_position_size = self.position_size_fixed_dollar / self.last_price


        # If entry position (long or short)
        if action != 0 and self.position == 0:
            self.entry_size = self.max_position_size
            # Buy entry
            if action == 1:
                self.long_trades += 1.0
                self.position = 1.0
                
                self.execution_price = self.last_price + self.last_price*self.transaction_fee_pct
                self.unrealized_pnl = self.position * (self.last_price - self.execution_price)*self.entry_size
                self.capital = self.initial_capital + self.realized_pnl + self.unrealized_pnl
            else: # Sell entry
                self.short_trades += 1.0
                self.position = -1.0
                self.execution_price = self.last_price - self.last_price*self.transaction_fee_pct
                self.unrealized_pnl = self.position * (self.last_price - self.execution_price)*self.entry_size
                self.capital = self.initial_capital + self.realized_pnl + self.unrealized_pnl
        # If exit position
        elif action != 0 and self.position != 0:
            entry_execution_price = self.execution_price
            # Buy exit
            prior_realized_pnl = self.realized_pnl
            if action == 1:
                
                
                self.execution_price = self.last_price + self.last_price*self.transaction_fee_pct
                self.realized_pnl += self.position * (self.execution_price - entry_execution_price)*self.entry_size
                self.capital = self.initial_capital + self.realized_pnl
                self.unrealized_pnl = 0.0
                if (self.realized_pnl - prior_realized_pnl) > 0:
                    self.short_trade_wins += 1.0
            else: # Sell exit
                
                self.execution_price = self.last_price - self.last_price*self.transaction_fee_pct
                self.realized_pnl += self.position * (self.execution_price - entry_execution_price)*self.entry_size
                self.capital = self.initial_capital + self.realized_pnl
                self.unrealized_pnl = 0.0
                if (self.realized_pnl - prior_realized_pnl) > 0:
                    self.long_trade_wins += 1.0
            self.position = 0.0
                
        # In existing position, no action (i.e. no update to said position)
        elif action == 0 and self.position != 0:
            self.unrealized_pnl = self.position * (self.last_price - self.execution_price)*self.entry_size
        # No existing position, no action
        elif action == 0 and self.position == 0:
            self.execution_price = np.nan
            self.unrealized_pnl = 0.0

        self.total_pnl = self.unrealized_pnl + self.realized_pnl

        self.overall_win_pct = (self.short_trade_wins + self.long_trade_wins) / (self.short_trades + self.long_trades + 1e-5)
        self.short_win_pct = self.short_trade_wins / (self.short_trades + 1e-5)
        self.long_win_pct = self.long_trade_wins / (self.long_trades + 1e-5)
        
        # Debug logging
        self.logger.debug(f"Before action {action}: Position={self.position}, Capital={self.capital}, " 
                         f"Realized PnL={self.realized_pnl}, Unrealized PnL={self.unrealized_pnl}")
        
        nav_change = self.capital - prior_nav
        self._cum_nav_chg += nav_change
        
        info = {
            "step": self.episode_step,
            "action": action,
            "price": self.last_price,
            "position":	self.position,
            "execution_price": self.execution_price,	
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "capital": self.capital,
            "nav_change": nav_change,
            "cumul_nav_change": self._cum_nav_chg,
            "long_win_pct": self.long_win_pct,
            "short_win_pct": self.short_win_pct,
            "win_pct": self.overall_win_pct, 
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "action_mask": self._get_action_mask()
        }
        
        return info
    
    def _calculate_reward(self, action: int, info: Dict[str, Any]) -> float:
        """
        Calculate the reward for the current step.
        
        Args:
            action: Agent action
            info: Action information
            
        Returns:
            Reward value
        """
        reward = self.reward_function.calculate_reward(
            action=action,
            position=self.position,
            position_change=info.get('position_change', 0.0),
            execution_price=info.get('execution_price', self.last_price),
            current_price=info.get('price', self.last_price),
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=info.get('realized_pnl_after', 0.0) - info.get('realized_pnl_before', 0.0),
            transaction_cost=info.get('transaction_cost', 0.0),
            info=info
        )
        
        self.episode_pnls.append({
            "step": self.episode_step,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "reward": reward
        })
        
        return reward
    
    def _is_episode_done(self) -> bool:
        """
        Check if the episode is done.
        
        Returns:
            True if done, False otherwise
        """
        # Episode is done if we've reached max steps
        if self.episode_step >= self.max_steps:
            return True
        
        # Episode is done if we've reached the end of the data in historical mode
        if self.mode == 'historical' and self.data is not None:
            if self.current_step >= len(self.data) - 1:
                return True
        
        # Episode is done if capital is depleted
        if self.capital <= 0:
            return True
        
        return False
    
    def _update_orderbook_historical(self, step_idx: int) -> None:
        """
        Update the orderbook with historical data for the given step.
        
        Args:
            step_idx: Index of the current step in the data
        """
        if step_idx < 0 or step_idx >= len(self.data):
            self.logger.error(f"Step index {step_idx} out of bounds for data length {len(self.data)}")
            return
        
        # Convert row of data to dictionary
        row_data = self.data[step_idx]
        orderbook_data = {}
        
        # Debug logging
        self.logger.debug(f"Data row at step {step_idx}: {row_data}")
        self.logger.debug(f"Feature columns: {self.feature_columns}")
        
        for i, col in enumerate(self.feature_columns):
            if i < len(row_data):
                orderbook_data[col] = row_data[i]
        
        # Debug logging
        self.logger.debug(f"Orderbook data: {orderbook_data}")
        # print(f"Orderbook data: {orderbook_data}")
        
        # Store the current state
        self.current_state = orderbook_data
        
        # Update orderbook with the data
        self.orderbook.update(orderbook_data)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Observation as numpy array
        """
        # Get orderbook state
        orderbook_state = self.orderbook.get_state_array()
        
        # Add position to observation
        observation = np.append(orderbook_state, [self.position])
        
        if self.stacked_obs is None:
            self.stacked_obs = np.repeat([observation], self.config.agents.observation_space.n_stack, axis=0)
        else:
            # WARNING: This assumes we don't call this function outside of one call in `step()`
            self.stacked_obs = np.append(self.stacked_obs[1:], [observation], axis=0)

        return self.stacked_obs.flatten()
    
    def _get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary of the episode.
        
        Returns:
            Dictionary with episode summary
        """
        return {
            "episode_length": self.episode_step,
            "final_position": self.position,
            "final_capital": self.capital,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "num_trades": len(self.episode_trades),
            "mean_reward": np.mean(self.episode_rewards),
            "total_reward": np.sum(self.episode_rewards),
            "transaction_costs": self.transaction_costs
        }
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            None
        """
        
        print(f"Step: {self.episode_step}/{self.max_steps}")
        print(f"Position: {self.position}")
        print(f"Capital: {self.capital:.2f}")
        print(f"Realized P&L: {self.realized_pnl:.2f}")
        print(f"Unrealized P&L: {self.unrealized_pnl:.2f}")
        print(f"Total P&L: {(self.realized_pnl + self.unrealized_pnl):.2f}")
        print(f"Transaction Costs: {self.transaction_costs:.2f}")
        
        if self.last_price is not None:
            print(f"Last Price: {self.last_price:.4f}")
        
        # Print bid-ask spread
        bid = self.orderbook.get_bid_price(1)
        ask = self.orderbook.get_ask_price(1)
        if bid is not None and ask is not None:
            print(f"Bid-Ask: {bid:.4f} - {ask:.4f} (Spread: {(ask - bid):.4f})")
        
        print("-" * 50)
    
    def close(self):
        """
        Close the environment.
        """
        pass

    @staticmethod
    def create_vectorized_envs(config: Dict[str, Any], data: np.ndarray, num_envs: int = 1) -> gym.vector.VectorEnv:
        """
        Create a vectorized environment.
        
        Args:
            config: Configuration dictionary
            data: Historical data
            num_envs: Number of environments to create
            
        Returns:
            Vectorized environment
        """
        # Define a function to create a single environment
        def make_env():
            return TradingEnvironment(config, data)
        
        # Create a vectorized environment
        envs = gym.vector.AsyncVectorEnv([lambda: make_env() for _ in range(num_envs)])
        
        return envs

    def _get_config_value(self, key_path: str, default_value: Any = None) -> Any:
        """
        Get a configuration value using dot notation to access nested dictionaries.
        
        Args:
            key_path: Path to the configuration value using dot notation (e.g., 'system.mode')
            default_value: Default value to return if the key is not found
            
        Returns:
            Configuration value or default value if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default_value
            return value
        except (KeyError, TypeError):
            return default_value