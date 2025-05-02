import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

class HistoricalDataReader:
    """
    Reads and processes historical order book data from CSV files.
    """
    def __init__(self, config: Dict[str, Any], split_num_for_dataset: int = None):
        """
        Initialize the historical data reader.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.data_path = self.config.data.historical.data_path if not split_num_for_dataset else f"{self.config.data.historical.data_path}/train_test_{split_num_for_dataset}.csv"
        self.feature_columns = self.config.data.historical.feature_columns 
        self.timestamp_column = self.config.data.historical.timestamp_column 
        self.normalize_features = self.config.data.historical.normalize_features 
        self.train_test_split = self.config.data.historical.train_test_split 
        self.price_column = self.config.data.historical.price_column
        self.atr_column = self.config.data.historical.atr_column
        
        self.data = None
        self.train_data = None
        self.test_data = None
        self.feature_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load order book data from CSV file.
        
        Returns:
            DataFrame containing order book data
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            # import pdb; pdb.set_trace()
            df = pd.read_csv(self.data_path)
            
            # Ensure timestamp column exists
            if self.timestamp_column not in df.columns:
                self.logger.warning(f"Timestamp column '{self.timestamp_column}' not found in data. Using default index.")
            else:
                # Convert timestamp to datetime and sort
                df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
                df = df.sort_values(by=self.timestamp_column)
            
            # Ensure all required feature columns exist
            missing_cols = [col for col in self.feature_columns if col not in df.columns]

            if missing_cols:
                self.logger.error(f"Missing required columns in data: {missing_cols}")
                raise ValueError(f"Missing required columns in data: {missing_cols}")
            
            self.data = df
            self.logger.info(f"Loaded {len(df)} rows of order book data")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise
    
    def preprocess_data(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess the loaded data for RL training.
        
        Returns:
            Tuple of preprocessed data as numpy array and feature statistics
        """
        if self.data is None:
            self.load_data()
        
        # Extract feature columns
        features_df = self.data[self.feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize features if required
        
        if self.normalize_features:
            self.logger.info("Normalizing features")
            for col in self.feature_columns:
                # import pdb; pdb.set_trace()
                mean = features_df[col].mean()
                std = features_df[col].std()
                features_df[col] = (features_df[col] - mean) / (std if std > 0 else 1)
                self.feature_stats[col] = {'mean': mean, 'std': std}
        
        # Convert to numpy array
        features = features_df.to_numpy()
        prices = self.data[self.price_column]
        atrs = self.data[self.atr_column]
        timestamps = self.data[self.timestamp_column]
        regimes = self.data['regime']
        
        # Split into train and test sets
        split_idx = int(len(features) * self.train_test_split)
        self.train_data = features[:split_idx]
        self.test_data = features[split_idx:]
        
        self.logger.info(f"Preprocessed data: {len(self.train_data)} training samples, {len(self.test_data)} testing samples")
        
        return features, prices, atrs, timestamps, regimes


    def preprocess_data_for_cv(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess the loaded data for RL training.
        
        Returns:
            Tuple of preprocessed data as numpy array and feature statistics
        """
        if self.data is None:
            self.load_data()
        
        # Extract feature columns
        features_df = self.data[self.feature_columns].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize features if required
        
        if self.normalize_features:
            self.logger.info("Normalizing features")
            for col in self.feature_columns:
                # import pdb; pdb.set_trace()
                mean = features_df[col].mean()
                std = features_df[col].std()
                features_df[col] = (features_df[col] - mean) / (std if std > 0 else 1)
                self.feature_stats[col] = {'mean': mean, 'std': std}
        
        # Convert to numpy array
        features = features_df.to_numpy()
        prices = self.data[self.price_column]
        atrs = self.data[self.atr_column]
        timestamps = self.data[self.timestamp_column]
        regimes =  self.data['regime']
        
        # Split into train and test sets
        split_idx = int(len(features) * self.train_test_split)
        train_states = features[:split_idx]
        test_states = features[split_idx:]

        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]

        train_atrs = atrs[:split_idx]
        test_atrs = atrs[split_idx:]

        train_timestamps = timestamps[:split_idx]
        test_timestamps = timestamps[split_idx:]

        train_regimes = regimes[:split_idx]
        test_regimes = regimes[split_idx:]
        
        return (train_states, train_prices, train_atrs, train_timestamps, train_regimes.reset_index(drop=True)), (test_states, test_prices.reset_index(drop=True), test_atrs.reset_index(drop=True), test_timestamps.reset_index(drop=True), test_regimes.reset_index(drop=True))
    
    def get_train_data(self) -> np.ndarray:
        """
        Get training data.
        
        Returns:
            Training data as numpy array
        """
        if self.train_data is None:
            self.preprocess_data()
        return self.train_data
    
    def get_test_data(self) -> np.ndarray:
        """
        Get testing data.
        
        Returns:
            Testing data as numpy array
        """
        if self.test_data is None:
            self.preprocess_data()
        return self.test_data
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature statistics used for normalization.
        
        Returns:
            Dictionary of feature statistics
        """
        if not self.feature_stats and self.normalize_features:
            self.preprocess_data()
        return self.feature_stats
