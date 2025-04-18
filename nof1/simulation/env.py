import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import uuid
from datetime import datetime
import random

from nof1.simulation.orderbook import OrderBook
from nof1.simulation.rewards import get_reward_function, RewardFunction

class TradingEnvironment(gym.Env):
    """
    Trading environment for RL agents.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict[str, Any], states: Optional[np.ndarray] = None, prices: Optional[np.ndarray] = None, atrs: Optional[np.ndarray] = None, timestamps: Optional[np.ndarray] = None):
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
        self.run_id = str(datetime.now())
        
        # System configuration
        self.mode = self.config.system.mode
        self.execution_price = None
        self.capital = None
        # Environment configuration
        self.max_steps = self.config.simulation.max_steps_per_episode
        self.warmup_steps = self.config.simulation.warmup_steps
        self.live_plot = self.config.simulation.live_plot

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
        
        self.states = states
        self.prices = prices 
        self.atrs = atrs
        self.timestamps = timestamps
        self.capital = 10000
        self.returns = [self.capital]
        self.episode_rewards = []
        self.pt_atr_mult = self.config.simulation.pt_atr_mult
        self.sl_atr_mult = self.config.simulation.sl_atr_mult
        self.reward_obj = get_reward_function(config)
        
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

        self._step = random.randint(1, len(self.states) - self.config.simulation.max_steps_per_episode) if self.config.simulation.random_start else 1
        self._starting_step = self._step
        
        self.position = 0
        self.current_state = np.append(self.states[self._step-1], [self.position])
        self.current_price = self.prices[self._step]
        self.atr = self.atrs[self._step-1]
        self.entry_price = None
        self.profit_target = None
        self.stop_loss = None
        self.trade_blotter = []
        self.capital = 10000
        self.returns = [self.capital]
        self.episode_rewards = []
        self.short_trade_wins = 0
        self.long_trade_wins = 0
        self.short_trades = 0
        self.long_trades = 0
        self._infos = []  # Reset the info history
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.reward_obj = None
        self.reward_obj = get_reward_function(self.config)
        self.num_trades = [0]

        
        # Initial info
        info = {
            "position": self.position,
            "capital": self.capital,
            "step": self._step,
            "action_mask": self._get_action_mask(),
            "price": self.current_price
        }
        self._infos.append(info)  # Store initial info
        
        return self.current_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Agent action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """        
        # Process the action
        info = self._process_action(action)

        self._step += 1
        
        # Get new observation
        self.current_state = np.append(self.states[self._step-1], [self.position]).astype(np.float32)
        self.current_price = self.prices[self._step]
        self.atr = self.atrs[self._step-1]
        
        
        # NOTE: Calculate reward
        reward = self._calculate_reward(info['step_return'], info)
        self.episode_rewards.append(reward)
        
        # Store info for plotting
        self._infos.append(info)
        
        # NOTE: Check if episode is done
        terminated = self._is_episode_done()
        # Truncated is used when episode is cut off prematurely (e.g., due to time limits)
        truncated = False

        if self.live_plot:
            self._create_returns_plot()
        
        
        # NOTE: 
        if terminated:
            info.update(self._get_episode_summary())
            
        return self.current_state, reward, terminated, truncated, info
    
    def _create_returns_plot(self):
        """
        Create a line chart that plots price and capital over time.
        Save the plot as 'live_returns.png'.
        """
        if not self._infos:
            self.logger.warning("No info data available for plotting")
            return
            
        # Extract data from info history
        steps = [info.get('step', i) for i, info in enumerate(self._infos)]
        prices = [info.get('price', 0) for info in self._infos]
        capitals = [info.get('capital', 0) for info in self._infos]
        
        # Create figure with two y-axes
        fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
        
        # Plot price on the first y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(steps, prices, color=color, label='Price')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create a second y-axis for capital
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Capital', color=color)
        ax2.plot(steps, capitals, color=color, label='Capital')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title and legend
        plt.title('Price and Capital over Time')
        
        # Add legends for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Adjust layout and save
        fig.tight_layout()
        plt.savefig('live_returns.png')
        plt.close(fig)
        
        self.logger.info("Saved returns plot to live_returns.png")
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Get a binary mask indicating which actions are valid in the current state.
        
        Returns:
            Binary mask where 1 = valid action, 0 = invalid action
        """
        # Initialize mask with all actions valid
        mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # If at max position, can't buy more
        if self.position > 0:
            mask[1] = 0  # Disable buy action
        
        # If at min position, can't sell more
        if self.position < 0:
            mask[2] = 0  # Disable sell action

        if self.position == 0:
            mask[3] = 0
            
        
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
            action = 0  # Convert to no-op (hold)

        # import pdb; pdb.set_trace()
        action_label = "NoOp"
        info = {}
        reset_internals = False
        trade_pnl = 0.0
        is_entry = False

        # If entry position (long or short)
        if action != 0 and self.position == 0:
            # Buy entry
            is_entry = True
            self.entry_price = self.current_price
            self.entry_step = self._step
            self.entry_time = self.timestamps[self._step]
            self.entry_state = self.current_state
            self.entry_state_step = self._step - 1
            self.next_entry_state = np.append(self.states[self._step] if self._step <= len(self.states) - 1 else self.states[self._step-1], [self.position])
            self.next_entry_state_step = self._step
            
            if any([np.isnan(s) for s in self.entry_state]):
                print(f"self.entry_state is nan!")
                import pdb; pdb.set_trace()

            
            if any([np.isnan(s) for s in self.next_entry_state]):
                print(f"self.entry_state is nan!")
                import pdb; pdb.set_trace()


            if action == 1:
                action_label = "LongEntry"
                self.long_trades += 1
                self.position = float(self.config.simulation.position_size_fixed_dollar / (self.sl_atr_mult*self.atr)) if self.config.simulation.allow_fractional_position_size else int(self.config.simulation.position_size_fixed_dollar / (self.sl_atr_mult*self.atr))
                self.profit_target = self.entry_price + self.pt_atr_mult*self.atr
                self.stop_loss = self.entry_price - self.sl_atr_mult*self.atr
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor
                self.unrealized_pnl = 0.0 - commish - slippage
                self.long_trades += 1
            else: # Sell entry
                action_label = "ShortEntry"
                self.short_trades += 1
                self.position = -float(self.config.simulation.position_size_fixed_dollar / (self.sl_atr_mult*self.atr)) if self.config.simulation.allow_fractional_position_size else -int(self.config.simulation.position_size_fixed_dollar / (self.sl_atr_mult*self.atr))
                self.profit_target = self.entry_price - self.pt_atr_mult*self.atr
                self.stop_loss = self.entry_price + self.sl_atr_mult*self.atr
                self.short_trades += 1
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor
                self.unrealized_pnl = 0.0 - commish - slippage

        # In existing position, no action (i.e. no update to said position)
        elif self.position > 0:
            # NOTE check if PT or SL and if yes --> check if PT or SL is met and close the position and calc PnL accordingly ELSE just update the unrealized pnl
            self.next_exit_state = np.append(self.states[self._step] if self._step <= len(self.states) - 1 else self.states[self._step], [self.position])
            self.next_exit_state_step = self._step

            if self.current_price >= self.profit_target:
                self.long_trade_wins += 1
                action_label = "LongExit"
                reset_internals = True
                trade_pnl = (self.profit_target - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": False, "commish": commish, "slippage": slippage, "entry_action": 1, "exit_action": 3, "entry_time": self.entry_time, "entry_state_step": self.entry_state_step, "entry_next_state_step": self.entry_state_step + 1, "entry_step": self.entry_step, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            elif self.current_price <= self.stop_loss:
                action_label = "LongExit"
                reset_internals = True
                trade_pnl = (self.stop_loss - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": False, "commish": commish, "slippage": slippage, "entry_action": 1, "exit_action": 3, "entry_time": self.entry_time, "entry_state_step": self.entry_state_step, "entry_next_state_step": self.entry_state_step + 1, "entry_step": self.entry_step, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            # NOTE: Check if agent closed the position
            elif action == 3:
                action_label = "LongExit"
                reset_internals = True
                trade_pnl = (self.current_price - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": False, "commish": commish, "slippage": slippage, "entry_action": 1, "exit_action": 3, "entry_time": self.entry_time, "entry_state_step": self.entry_state_step, "entry_next_state_step": self.entry_state_step + 1, "entry_step": self.entry_step, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            else: 
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor
                self.unrealized_pnl = ((self.current_price - self.entry_price)*self.position) - commish - slippage

        elif self.position < 0:
            # NOTE check if PT or SL and if yes --> check if PT or SL is met and close the position and calc PnL accordingly ELSE just update the unrealized pnl
            self.next_exit_state = np.append(self.states[self._step] if self._step <= len(self.states) - 1 else self.states[self._step], [self.position])
            self.next_exit_state_step = self._step
            if self.current_price <= self.profit_target:
                self.short_trade_wins += 1
                action_label = "ShortExit"
                reset_internals = True
                trade_pnl = (self.profit_target - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2 
                self.realized_pnl += trade_pnl - commish - slippage
                self.unrealized_pnl = 0.0
                # import pdb; pdb.set_trace()
                self.trade_blotter.append({"terminal": False, "commish": commish, "slippage": slippage, "entry_action": 2, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "exit_time": self.timestamps[self._step], "quantity": abs(self.position)})
            elif self.current_price >= self.stop_loss:
                action_label = "ShortExit"
                reset_internals = True
                trade_pnl = (self.stop_loss - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2 
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                self.unrealized_pnl = 0.0
                # import pdb; pdb.set_trace()
                self.trade_blotter.append({"terminal": False, "commish": commish, "slippage": slippage, "entry_action": 2, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            # NOTE: Check if agent closed the position
            elif action == 3:
                action_label = "ShortExit"
                reset_internals = True
                trade_pnl = (self.current_price - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2 
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": False, "commish": commish, "slippage": slippage, "entry_action": 2, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            else:
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct 
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor
                self.unrealized_pnl = ((self.current_price - self.entry_price)*self.position) - commish - slippage

        self.capital = self.initial_capital + self.unrealized_pnl + self.realized_pnl
        self.returns.append(self.capital)

        step_return = self.returns[-1] - self.returns[-2]
        info['step_return'] = step_return
        

        if reset_internals:
            self.profit_target = None
            self.stop_loss = None
            self.entry_price = None
            self.entry_step = None
            self.position = 0


        self.overall_win_pct = (self.short_trade_wins + self.long_trade_wins) / (self.short_trades + self.long_trades + 1e-5)
        self.short_win_pct = self.short_trade_wins / (self.short_trades + 1e-5)
        self.long_win_pct = self.long_trade_wins / (self.long_trades + 1e-5)
        
        # Debug logging
        self.logger.debug(f"Before action {action}: Position={self.position}, Capital={self.capital}")

        if is_entry:
            self.num_trades.append(self.long_trades+self.short_trades)
        
        info = {
            "step": self._step,
            "action": action,
            "price": self.current_price,
            "position":	self.position,
            "capital": self.capital,
            "nav_change": step_return,
            "step_return": step_return,
            "long_win_pct": self.long_win_pct,
            "short_win_pct": self.short_win_pct,
            "win_pct": self.overall_win_pct, 
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "action_mask": self._get_action_mask(),
            "action_label": action_label,
            "num_trades_delta": self.num_trades[-1] - self.num_trades[-2] if len(self.num_trades) > 1 else self.num_trades[-1]
        }
        return info
    
    def _calculate_reward(self, reward: int, info: Dict[str, Any]) -> float:
        """
        Calculate the reward for the current step.
        
        Args:
            action: Agent action
            info: Action information
            
        Returns:
            Reward value
        """
        
        reward = self.reward_obj.calculate_reward(reward, info)
        
        return reward
    
    def _is_episode_done(self) -> bool:
        """
        Check if the episode is done.
        
        Returns:
            True if done, False otherwise
        """
        # Episode is done if we've reached max steps
        if self._step >= self.max_steps and self._starting_step < self.max_steps:
            self.next_exit_state = np.append(self.states[self._step] if self._step <= len(self.states) - 1 else self.states[self._step], [self.position])
            self.next_exit_state_step = self._step
            if self.position > 0:
                trade_pnl = (self.profit_target - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                if trade_pnl > 0:
                    self.long_trade_wins += 1
            
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": True, "commish": commish, "slippage": slippage, "entry_action": 1, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_state_step": self.entry_state_step, "next_entry_state_step": self.entry_state_step + 1, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            elif self.position < 0:
                trade_pnl = (self.current_price - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                
                if trade_pnl > 0:
                    self.short_trade_wins += 1
            
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": True, "commish": commish, "slippage": slippage, "entry_action": 2, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_state_step": self.entry_state_step, "next_entry_state_step": self.entry_state_step + 1, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            
            self.overall_win_pct = (self.short_trade_wins + self.long_trade_wins) / (self.short_trades + self.long_trades + 1e-5)
            self.short_win_pct = self.short_trade_wins / (self.short_trades + 1e-5)
            self.long_win_pct = self.long_trade_wins / (self.long_trades + 1e-5)

            self.capital = self.initial_capital + self.unrealized_pnl + self.realized_pnl
            self.returns.append(self.capital)
            
            
            df = pd.DataFrame.from_records(self.trade_blotter)
            if len(df) > 0:
                episode_hash = uuid.uuid4().hex
                df['entry_time'] = df['entry_time'].astype(str)
                df['exit_time'] = df['exit_time'].astype(str)
                df["episode_id"] = [episode_hash]*len(df)
                df.to_csv(f"./results/trade_blotter_{self.run_id}.csv", mode='a', header=not os.path.exists(f"./results/trade_blotter_{self.run_id}.csv"), index=False)
                
                df_infos = pd.DataFrame.from_records(self._infos)
                df_infos["episode_id"] = [episode_hash]*len(df_infos)
                df_infos.to_csv(f"./results/trade_stats_{self.run_id}.csv", mode='a', header=not os.path.exists(f"./results/trade_stats_{self.run_id}.csv"), index=False)

            return True
        
        # Episode is done if we've reached the end of the data in historical mode
        if self.mode == 'historical' and self.states is not None:
            self.next_exit_state = np.append(self.states[self._step] if self._step <= len(self.states) - 1 else self.states[self._step], [self.position])
            self.next_exit_state_step = self._step
            if self._step >= len(self.states) - 1:
                if self.position > 0:
                    trade_pnl = (self.profit_target - self.entry_price)*self.position
                    commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                    slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                    self.realized_pnl += trade_pnl - commish - slippage
                    if trade_pnl > 0:
                        self.long_trade_wins += 1
                
                    self.unrealized_pnl = 0.0
                    self.trade_blotter.append({"terminal": True, "commish": commish, "slippage": slippage, "entry_action": 1, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_state_step": self.entry_state_step, "next_entry_state_step": self.entry_state_step + 1, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
                elif self.position < 0:
                    trade_pnl = (self.current_price - self.entry_price)*self.position
                    commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                    slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                    self.realized_pnl += trade_pnl - commish
                    if trade_pnl > 0:
                        self.short_trade_wins += 1
                
                    self.unrealized_pnl = 0.0
                    self.trade_blotter.append({"terminal": True, "commish": commish,  "slippage": slippage, "entry_action": 2, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_state_step": self.entry_state_step, "next_entry_state_step": self.entry_state_step + 1, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
                
                self.overall_win_pct = (self.short_trade_wins + self.long_trade_wins) / (self.short_trades + self.long_trades + 1e-5)
                self.short_win_pct = self.short_trade_wins / (self.short_trades + 1e-5)
                self.long_win_pct = self.long_trade_wins / (self.long_trades + 1e-5)

                self.capital = self.initial_capital + self.unrealized_pnl + self.realized_pnl
                self.returns.append(self.capital)
                
                
                df = pd.DataFrame.from_records(self.trade_blotter)
                if len(df) > 0:
                    episode_hash = uuid.uuid4().hex
                    df['entry_time'] = df['entry_time'].astype(str)
                    df['exit_time'] = df['exit_time'].astype(str)
                    df["episode_id"] = [episode_hash]*len(df)
                    df.to_csv(f"./results/trade_blotter_{self.run_id}.csv", mode='a', header=not os.path.exists(f"./results/trade_blotter_{self.run_id}.csv"), index=False)
                    
                    df_infos = pd.DataFrame.from_records(self._infos)
                    df_infos["episode_id"] = [episode_hash]*len(df_infos)
                    df_infos.to_csv(f"./results/trade_stats_{self.run_id}.csv", mode='a', header=not os.path.exists(f"./results/trade_stats_{self.run_id}.csv"), index=False)
                
                return True

        # Episode is done if capital is depleted
        if self.capital <= 0:
            self.next_exit_state = np.append(self.states[self._step] if self._step <= len(self.states) - 1 else self.states[self._step], [self.position])
            self.next_exit_state_step = self._step
            if self.position > 0:
                trade_pnl = (self.profit_target - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                if trade_pnl > 0:
                    self.long_trade_wins += 1
            
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": True, "commish": commish, "slippage": slippage, "entry_action": 1, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_state_step": self.entry_state_step, "next_entry_state_step": self.entry_state_step + 1, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            elif self.position < 0:
                trade_pnl = (self.current_price - self.entry_price)*self.position
                commish = self.entry_price*abs(self.position)*self.config.simulation.transaction_fee_pct*2
                slippage = self.entry_price*abs(self.position)*self.config.simulation.slippage_factor*2
                self.realized_pnl += trade_pnl - commish - slippage
                if trade_pnl > 0:
                    self.short_trade_wins += 1
            
                self.unrealized_pnl = 0.0
                self.trade_blotter.append({"terminal": True, "commish": commish, "slippage": slippage, "entry_action": 2, "exit_action": 3, "entry_time": self.entry_time, "entry_step": self.entry_step, "entry_state_step": self.entry_state_step, "next_entry_state_step": self.entry_state_step + 1, "entry_price": self.entry_price, "exit_step": self._step, "exit_price": self.current_price, "exit_time": self.timestamps[self._step], "pnl": trade_pnl, "exit_state_step": self._step-1, "next_exit_state_step": self.next_exit_state_step, "quantity": abs(self.position)})
            
            self.overall_win_pct = (self.short_trade_wins + self.long_trade_wins) / (self.short_trades + self.long_trades + 1e-5)
            self.short_win_pct = self.short_trade_wins / (self.short_trades + 1e-5)
            self.long_win_pct = self.long_trade_wins / (self.long_trades + 1e-5)

            self.capital = self.initial_capital + self.unrealized_pnl + self.realized_pnl
            self.returns.append(self.capital)
            
            df = pd.DataFrame.from_records(self.trade_blotter)
            if len(df) > 0:
                episode_hash = uuid.uuid4().hex
                df['entry_time'] = df['entry_time'].astype(str)
                df['exit_time'] = df['exit_time'].astype(str)
                df["episode_id"] = [episode_hash]*len(df)
                df.to_csv(f"./results/trade_blotter_{self.run_id}.csv", mode='a', header=not os.path.exists(f"./results/trade_blotter_{self.run_id}.csv"), index=False)
                
                df_infos = pd.DataFrame.from_records(self._infos)
                df_infos["episode_id"] = [episode_hash]*len(df_infos)
                df_infos.to_csv(f"./results/trade_stats_{self.run_id}.csv", mode='a', header=not os.path.exists(f"./results/trade_stats_{self.run_id}.csv"), index=False)
            
            return True
        
        return False
    
    def _get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary of the episode.
        
        Returns:
            Dictionary with episode summary
        """
        return {
            "episode_length": self._step,
            "final_position": self.position,
            "final_capital": self.capital,
            "num_trades": len(self.trade_blotter),
            "mean_reward": np.mean(self.episode_rewards),
            "total_reward": np.sum(self.episode_rewards)
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

    def rollout(self, policy, batch_size=None, n_steps=100):
        """
        Perform a rollout using the given policy.
        
        Args:
            policy: Policy network that maps state to actions
            batch_size: Batch size for vectorized execution
            n_steps: Number of steps in the rollout
            
        Returns:
            total_rewards: Total rewards accumulated during rollout
            all_states: List of states visited
            all_actions: List of actions taken
            all_rewards: List of rewards received
        """
        # Reset environment - this will initialize different environments for each batch element
        obs, info = self.reset()
        
        # If policy has a reset method (e.g., for RNNs), reset it with the correct batch size
        if hasattr(policy, 'reset'):
            if batch_size:
                policy.reset(batch_size[0] if isinstance(batch_size, list) else batch_size)
            else:
                policy.reset()
        
        # Initialize lists to store trajectory
        all_obs = [obs]
        all_actions = []
        all_rewards = []
        
        # Accumulate total reward
        total_rewards = torch.zeros(batch_size if batch_size else ())
        
        # Perform rollout
        for _ in range(n_steps):
            # Get action from policy
            action = policy(obs)
            all_actions.append(action)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.step(action)
            all_rewards.append(reward)
            
            # Accumulate rewards
            total_rewards += reward
            
            # Update state
            obs = next_obs
            all_obs.append(obs)
        
        return total_rewards, all_obs, all_actions, all_rewards