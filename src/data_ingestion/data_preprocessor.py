import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional

class DataPreprocessor:
    """
    Preprocesses order book data for the RL environment.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.feature_columns = config.get('data.historical.feature_columns', [])
        self.normalize_features = config.get('data.historical.normalize_features', True)
        self.feature_stats = {}
        
    def compute_feature_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics (mean, std) for each feature column.
        
        Args:
            data: DataFrame containing feature columns
            
        Returns:
            Dictionary of feature statistics
        """
        feature_stats = {}
        
        for col in self.feature_columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                feature_stats[col] = {'mean': mean, 'std': std}
            else:
                self.logger.warning(f"Feature column '{col}' not found in data")
        
        self.feature_stats = feature_stats
        return feature_stats
    
    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using the computed statistics.
        
        Args:
            data: DataFrame containing feature columns
            
        Returns:
            DataFrame with normalized features
        """
        if not self.feature_stats:
            self.compute_feature_stats(data)
        
        normalized_df = data.copy()
        
        for col in self.feature_columns:
            if col in normalized_df.columns and col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                normalized_df[col] = (normalized_df[col] - mean) / (std if std > 0 else 1.0)
        
        return normalized_df
    
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize a single observation using the computed statistics.
        
        Args:
            observation: Raw observation array
            
        Returns:
            Normalized observation array
        """
        if not self.feature_stats or len(observation) != len(self.feature_columns):
            return observation
        
        normalized_obs = np.zeros_like(observation)
        
        for i, col in enumerate(self.feature_columns):
            if col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                normalized_obs[i] = (observation[i] - mean) / (std if std > 0 else 1.0)
            else:
                normalized_obs[i] = observation[i]
        
        return normalized_obs
    
    def denormalize_observation(self, normalized_obs: np.ndarray) -> np.ndarray:
        """
        Denormalize a normalized observation back to original scale.
        
        Args:
            normalized_obs: Normalized observation array
            
        Returns:
            Denormalized observation array
        """
        if not self.feature_stats or len(normalized_obs) != len(self.feature_columns):
            return normalized_obs
        
        denormalized_obs = np.zeros_like(normalized_obs)
        
        for i, col in enumerate(self.feature_columns):
            if col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                denormalized_obs[i] = normalized_obs[i] * (std if std > 0 else 1.0) + mean
            else:
                denormalized_obs[i] = normalized_obs[i]
        
        return denormalized_obs
    
    def extract_features(self, orderbook_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from order book data.
        
        Args:
            orderbook_data: Dictionary containing order book data
            
        Returns:
            Feature array
        """
        features = []
        
        for col in self.feature_columns:
            if col in orderbook_data:
                features.append(orderbook_data[col])
            else:
                # If feature is missing, use 0 (log warning)
                self.logger.warning(f"Feature '{col}' missing in order book data")
                features.append(0.0)
        
        return np.array(features)
    
    def compute_derived_features(self, orderbook_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute derived features from order book data.
        
        Args:
            orderbook_data: Dictionary containing order book data
            
        Returns:
            Dictionary of derived features
        """
        derived_features = {}
        
        # Example derived features
        if 'bid_price_1' in orderbook_data and 'ask_price_1' in orderbook_data:
            # Bid-ask spread
            bid_price = orderbook_data['bid_price_1']
            ask_price = orderbook_data['ask_price_1']
            spread = ask_price - bid_price
            derived_features['spread'] = spread
            
            # Mid price
            mid_price = (bid_price + ask_price) / 2
            derived_features['mid_price'] = mid_price
            
            # Relative spread (%)
            if mid_price > 0:
                derived_features['relative_spread'] = 100 * spread / mid_price
        
        # Imbalance
        if 'bid_size_1' in orderbook_data and 'ask_size_1' in orderbook_data:
            bid_size = orderbook_data['bid_size_1']
            ask_size = orderbook_data['ask_size_1']
            total_size = bid_size + ask_size
            
            if total_size > 0:
                derived_features['order_imbalance'] = (bid_size - ask_size) / total_size
        
        return derived_features