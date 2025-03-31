import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from hyperliquid.info import Info
from hyperliquid.utils import constants


class HyperLiquidConnector:
    """
    Connector for processing HyperLiquid L2 book data from a local JSON file.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the HyperLiquid connector.
        
        Args:
            file_path: Path to the local JSON file containing L2 book data.
        """
        self.file_path = file_path
        self.data = None#self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from the JSON file.
        
        Returns:
            List of dictionaries containing the L2 book data.
        """
        with open(self.file_path, 'r') as f:
            # Each line is a separate JSON object
            return [json.loads(line) for line in f]
    
    def _process_levels(self, levels: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process the levels data by prepending 'bid_' and 'ask_' to keys.
        
        Args:
            levels: The levels data from the L2 book.
            
        Returns:
            Dictionary with processed bid and ask data.
        """
        # Process bids (first list)
        bids = levels[0]
        # Sort bids by price in descending order
        bids_sorted = sorted(bids, key=lambda x: float(x['px']), reverse=True)
        # Take top 5 bids
        bids_top10 = bids_sorted[:5]
        
        # Process asks (second list)
        asks = levels[1]
        # Sort asks by price in ascending order
        asks_sorted = sorted(asks, key=lambda x: float(x['px']))
        # Take bottom 5 asks
        asks_bottom10 = asks_sorted[:5]
        
        # Prepare data for DataFrame
        result = {}
        
        # Process bids
        for i, bid in enumerate(bids_top10):
            for key, value in bid.items():
                column_name = f"bid_{key}_{i}"
                result[column_name] = value  # Store the value directly, not as a list
        
        # Process asks
        for i, ask in enumerate(asks_bottom10):
            for key, value in ask.items():
                column_name = f"ask_{key}_{i}"
                result[column_name] = value  # Store the value directly, not as a list
                
        return result
    
    def fetch_data(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, symbol: str = "BTC") -> pd.DataFrame:
        """
        Fetch and process data from the loaded JSON file.
        
        Args:
            start_time: Optional start time filter. If provided, data for the entire day (24 hours) will be fetched.
            end_time: Optional end time filter.
            symbol: Trading symbol to fetch data for (default: "BTC")
            
        Returns:
            DataFrame with processed L2 book data.
        """
        processed_data = []
        seen_minutes = set()  # Track timestamps rounded to the minute

        # Download and process data for all 24 hours if start_time is provided
        if start_time:
            self._download_and_load_hourly_data(start_time, symbol)
        
        # If start_time is provided, set it to the beginning of the day and create an end_time for 24 hours later
        if start_time:
            # Set to beginning of the day (00:00:00)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            # If end_time is not explicitly provided, set it to the end of the same day
            if not end_time:
                end_time = start_time.replace(hour=23, minute=59, second=59)
        
        for entry in self.data:
            
            entry_time = datetime.fromisoformat(entry['time'])
            
            # Round to the nearest minute to reduce data volume but maintain reasonable granularity
            rounded_time = entry_time.replace(second=0, microsecond=0)
            rounded_time_str = rounded_time.strftime("%Y-%m-%d %H:%M:%S")  # Format as "YYYY-MM-DD HH:MM:SS"
            
            # Skip if we've already seen this minute
            if rounded_time_str in seen_minutes:
                continue
            
            # Apply time filters if provided
            if start_time and entry_time < start_time:
                continue
            if end_time and entry_time > end_time:
                continue
            
            # Check if this is an L2 book entry
            if entry['raw']['channel'] == 'l2Book':
                l2_data = entry['raw']['data']
                
                # Process the levels
                processed_levels = self._process_levels(l2_data['levels'])
                
                # Create a row with timestamp and market time
                row_data = {
                    'timestamp': rounded_time_str,  # Use rounded timestamp with new format
                    'market_time': l2_data['time'],
                    'coin': l2_data['coin']
                }
                
                # Add the processed levels data
                row_data.update(processed_levels)
                
                processed_data.append(row_data)
                seen_minutes.add(rounded_time_str)  # Mark this minute as seen
        
        # Create DataFrame
        if processed_data:
            # import pdb; pdb.set_trace()
            return pd.DataFrame(processed_data)
        
        else:
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """
        Get the list of unique symbols (coins) in the data.
        
        Returns:
            List of unique symbols.
        """
        symbols = set()
        for entry in self.data:
            if 'raw' in entry and 'data' in entry['raw'] and 'coin' in entry['raw']['data']:
                symbols.add(entry['raw']['data']['coin'])
        return list(symbols)

    def _download_and_load_hourly_data(self, date: datetime, symbol: str = "BTC") -> None:
        """
        Download and load data for all 24 hours of a given date.
        
        Args:
            date: The date to download data for
            symbol: Trading symbol to fetch data for
        """
        import os
        import json
        
        # Format date as YYYYMMDD for the S3 path
        date_str = date.strftime("%Y%m%d")
        all_data = []
        
        for hour in range(24):
            # Download the file from S3
            os.system(f"aws s3 cp s3://hyperliquid-archive/market_data/{date_str}/{hour}/l2Book/{symbol}.lz4 ./{symbol}_{hour}.lz4 --request-payer requester")
            os.system(f"unlz4 --rm ./{symbol}_{hour}.lz4")
            
            # Read the file
            self.file_path = f"./{symbol}_{hour}"
            try:
                with open(self.file_path, 'r') as f:
                    # Each line is a separate JSON object
                    hour_data = [json.loads(line) for line in f]
                    all_data.extend(hour_data)
                
                # Clean up the file
                os.remove(self.file_path)
            except FileNotFoundError:
                print(f"Warning: File not found for hour {hour}")
            except Exception as e:
                print(f"Error processing hour {hour}: {e}")
        
        # Update self.data with all the loaded data
        self.data = all_data


# Example usage
# if __name__ == "__main__":
#     import os
#     SYMBOL = "BTC"
#     FROM_DATE = "20250228"
#     HOUR = "9"
#     os.system(f"aws s3 cp s3://hyperliquid-archive/market_data/{FROM_DATE}/{HOUR}/l2Book/{SYMBOL}.lz4 ./{SYMBOL}.lz4 --request-payer requester")
#     os.system(f"unlz4 --rm ./{SYMBOL}.lz4")
    
#     connector = HyperLiquidConnector(f"./{SYMBOL}")
#     df = connector.fetch_data()
#     print(df.head())
#     print(f"Available symbols: {connector.get_symbols()}")

