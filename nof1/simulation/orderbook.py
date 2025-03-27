import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

class OrderBook:
    """
    Represents and manages the state of the order book.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the order book.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        
        self.tick_size = self.config.simulation.tick_size 
        self.slippage_model = self.config.simulation.slippage_model 
        self.slippage_value = self.config.simulation.slippage_value 
        
        # Current state of the order book
        self.current_state = {}
        
        # Access nested config values properly
        self.feature_columns = self.config.data.historical.feature_columns 
        print(f"self.feature_columns: {self.feature_columns}")
        
        # Track last trade price
        self.last_trade_price = None
    
    def update(self, orderbook_data: Dict[str, Any]) -> None:
        """
        Update the state of the order book with new data.
        
        Args:
            orderbook_data: Dictionary containing order book data
        """
        self.current_state = orderbook_data
        
        # Update last trade price to mid price if not set
        if self.last_trade_price is None and 'bid_price_1' in orderbook_data and 'ask_price_1' in orderbook_data:
            self.last_trade_price = (orderbook_data['bid_price_1'] + orderbook_data['ask_price_1']) / 2
    
    def get_mid_price(self) -> Optional[float]:
        """
        Get the current mid price.
        
        Returns:
            Mid price or None if not available
        """

        if 'bid_price_1' in self.current_state and 'ask_price_1' in self.current_state:
            return (self.current_state['bid_price_1'] + self.current_state['ask_price_1']) / 2
        return None
    
    def get_bid_price(self, level: int = 1) -> Optional[float]:
        """
        Get the current bid price at the specified level.
        
        Args:
            level: Order book level (1 = top of book)
            
        Returns:
            Bid price or None if not available
        """
        bid_key = f"bid_price_{level}"
        return self.current_state.get(bid_key)
    
    def get_ask_price(self, level: int = 1) -> Optional[float]:
        """
        Get the current ask price at the specified level.
        
        Args:
            level: Order book level (1 = top of book)
            
        Returns:
            Ask price or None if not available
        """
        ask_key = f"ask_price_{level}"
        return self.current_state.get(ask_key)
    
    def get_bid_size(self, level: int = 1) -> Optional[float]:
        """
        Get the current bid size at the specified level.
        
        Args:
            level: Order book level (1 = top of book)
            
        Returns:
            Bid size or None if not available
        """
        bid_key = f"bid_size_{level}"
        return self.current_state.get(bid_key)
    
    def get_ask_size(self, level: int = 1) -> Optional[float]:
        """
        Get the current ask size at the specified level.
        
        Args:
            level: Order book level (1 = top of book)
            
        Returns:
            Ask size or None if not available
        """
        ask_key = f"ask_size_{level}"
        return self.current_state.get(ask_key)
    
    def get_spread(self) -> Optional[float]:
        """
        Get the current bid-ask spread.
        
        Returns:
            Spread or None if not available
        """
        bid = self.get_bid_price(1)
        ask = self.get_ask_price(1)
        
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    def get_relative_spread(self) -> Optional[float]:
        """
        Get the current relative bid-ask spread (spread / mid_price).
        
        Returns:
            Relative spread or None if not available
        """
        spread = self.get_spread()
        mid_price = self.get_mid_price()
        
        if spread is not None and mid_price is not None and mid_price > 0:
            return spread / mid_price
        return None
    
    def get_order_imbalance(self) -> Optional[float]:
        """
        Get the current order imbalance (bid_size - ask_size) / (bid_size + ask_size).
        
        Returns:
            Order imbalance or None if not available
        """
        bid_size = self.get_bid_size(1)
        ask_size = self.get_ask_size(1)
        
        if bid_size is not None and ask_size is not None:
            total_size = bid_size + ask_size
            if total_size > 0:
                return (bid_size - ask_size) / total_size
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the order book.
        
        Returns:
            Dictionary containing order book state
        """
        print(f"self.current_state: {self.current_state}")
        return self.current_state.copy()
    
    def get_state_array(self) -> np.ndarray:
        """
        Get the current state of the order book as a numpy array.
        
        Returns:
            Numpy array containing order book state
        """
        state_array = []
        
        for col in self.feature_columns:
            if col in self.current_state:
                state_array.append(self.current_state[col])
            else:
                state_array.append(0.0)
        
        return np.array(state_array)
    
    def calculate_execution_price(self, action_type: str, size: float) -> Tuple[float, float]:
        """
        Calculate the execution price for a trade, considering slippage.
        
        Args:
            action_type: 'buy' or 'sell'
            size: Trade size
            
        Returns:
            Tuple of (execution_price, slippage)
        """
        if action_type == 'buy':
            base_price = self.get_ask_price(1)
            if base_price is None:
                base_price = self.get_mid_price()
        else:  # 'sell'
            base_price = self.get_bid_price(1)
            if base_price is None:
                base_price = self.get_mid_price()
        
        if base_price is None:
            self.logger.warning("Unable to determine base price for execution")
            base_price = self.last_trade_price
        
        # Calculate slippage based on the configured model
        slippage = 0.0
        
        if self.slippage_model == 'fixed':
            # Fixed slippage as percentage of price
            slippage = base_price * self.slippage_value
            
        elif self.slippage_model == 'proportional':
            # Slippage proportional to trade size and current liquidity
            if action_type == 'buy':
                ask_size = self.get_ask_size(1) or 1.0
                slippage = base_price * self.slippage_value * (size / ask_size)
            else:  # 'sell'
                bid_size = self.get_bid_size(1) or 1.0
                slippage = base_price * self.slippage_value * (size / bid_size)
        
        # Apply slippage in the correct direction
        if action_type == 'buy':
            execution_price = base_price + slippage
        else:  # 'sell'
            execution_price = base_price - slippage
        
        # Update last trade price
        self.last_trade_price = execution_price
        
        return execution_price, slippage