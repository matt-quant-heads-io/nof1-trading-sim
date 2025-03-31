import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Optional, Union, Dict, Any

class DataCollator:
    """
    A class to collate data from multiple sources (Polygon and Hyperliquid).
    
    This class fetches data from both sources and merges them based on timestamp.
    """
    
    def __init__(self, polygon_connector, hyperliquid_connector):
        """
        Initialize the DataCollator with connectors for different data sources.
        
        Args:
            polygon_connector: An instance of the Polygon connector
            hyperliquid_connector: An instance of the Hyperliquid connector
        """
        self.polygon_connector = polygon_connector
        self.hyperliquid_connector = hyperliquid_connector
    
    def get_merged_data(self, 
                        symbol: str, 
                        from_date: Union[str, datetime],
                        to_date: Optional[Union[str, datetime]] = None,
                        timespan: str = "minute",
                        timezone: str = "America/New_York") -> pd.DataFrame:
        """
        Fetch data from both Polygon and Hyperliquid and merge them.
        
        Args:
            symbol: The trading symbol (e.g., 'BTC')
            from_date: Start date for data fetching (string 'YYYY-MM-DD' or datetime)
            to_date: End date for data fetching (string 'YYYY-MM-DD' or datetime, optional)
            timespan: Time interval for the data (default: 'minute')
            timezone: Timezone for timestamp conversion (default: 'America/New_York')
            
        Returns:
            A pandas DataFrame with merged data from both sources
        """
        # Convert string dates to datetime if needed
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, "%Y-%m-%d")
        
        if to_date is None:
            to_date = datetime.now()
        elif isinstance(to_date, str):
            to_date = datetime.strptime(to_date, "%Y-%m-%d")
        
        # Get data from Hyperliquid
        hyperliquid_data = self.hyperliquid_connector.fetch_data(
            start_time=from_date,
            end_time=to_date
        )
        
        # Get data from Polygon
        polygon_data = self.polygon_connector.fetch_ohlc(
            timespan=timespan,
            multiplier=1,
            from_date=from_date,
            to_date=to_date
        )
        
        # Merge the datasets
        merged_data = self._merge_datasets(hyperliquid_data, polygon_data, timezone)
        
        return merged_data
    
    def _merge_datasets(self, 
                       hyperliquid_data: pd.DataFrame, 
                       polygon_data: pd.DataFrame,
                       timezone: str = "America/New_York") -> pd.DataFrame:
        """
        Merge datasets from Hyperliquid, Polygon, and YFinance based on timestamp.
        
        Args:
            hyperliquid_data: DataFrame containing Hyperliquid data
            polygon_data: DataFrame containing Polygon data
            yfinance_data: DataFrame containing YFinance data
            timezone: Timezone for timestamp conversion
            
        Returns:
            A merged DataFrame with data from all sources
        """
        # Ensure timestamps are in the same format and timezone
        if 'timestamp' in hyperliquid_data.columns:
            hyperliquid_time_col = 'timestamp'
        elif 't' in hyperliquid_data.columns:
            hyperliquid_time_col = 't'
        else:
            raise ValueError("Could not find timestamp column in Hyperliquid data")
        
        if 'timestamp' in polygon_data.columns:
            polygon_time_col = 'timestamp'
        elif 't' in polygon_data.columns:
            polygon_time_col = 't'
        else:
            raise ValueError("Could not find timestamp column in Polygon data")
        
        # Convert timestamps to datetime objects in the same timezone
        tz = pytz.timezone(timezone)
        
        # Process Hyperliquid timestamps
        if hyperliquid_data[hyperliquid_time_col].dtype == 'int64':
            # If timestamp is in milliseconds
            hyperliquid_data['timestamp'] = pd.to_datetime(
                hyperliquid_data[hyperliquid_time_col], unit='ms'
            ).dt.tz_localize('UTC').dt.tz_convert(tz)
        else:
            hyperliquid_data['timestamp'] = pd.to_datetime(
                hyperliquid_data[hyperliquid_time_col]
            ).dt.tz_localize('UTC').dt.tz_convert(tz)
        
        # Process Polygon timestamps
        if polygon_data[polygon_time_col].dtype == 'int64':
            # If timestamp is in milliseconds
            polygon_data['timestamp'] = pd.to_datetime(
                polygon_data[polygon_time_col], unit='ms'
            ).dt.tz_localize('UTC').dt.tz_convert(tz)
        else:
            polygon_data['timestamp'] = pd.to_datetime(
                polygon_data[polygon_time_col]
            ).dt.tz_localize('UTC').dt.tz_convert(tz)
        
        # Rename columns to avoid conflicts
        hyperliquid_cols = {col: f"hl_{col}" for col in hyperliquid_data.columns 
                           if col != 'timestamp' and col != hyperliquid_time_col}
        polygon_cols = {col: f"poly_{col}" for col in polygon_data.columns 
                       if col != 'timestamp' and col != polygon_time_col}
        # yfinance_cols = {col: f"vix_{col}" for col in yfinance_data.columns
        #                 if col != 'timestamp' and col != yfinance_time_col}
        
        hyperliquid_data = hyperliquid_data.rename(columns=hyperliquid_cols)
        polygon_data = polygon_data.rename(columns=polygon_cols)
        # yfinance_data = yfinance_data.rename(columns=yfinance_cols)
        
        # Merge on timestamp
        merged_data = pd.merge(
            hyperliquid_data[['timestamp'] + list(hyperliquid_cols.values())],
            polygon_data[['timestamp'] + list(polygon_cols.values())],
            on='timestamp',
            how='inner'
        )
        
        # Sort by timestamp
        merged_data = merged_data.sort_values('timestamp')
        
        return merged_data
    
    def get_latest_data(self, 
                       symbol: str, 
                       lookback_days: int = 1,
                       timespan: str = "minute",
                       timezone: str = "America/New_York") -> pd.DataFrame:
        """
        Fetch the most recent data from both sources.
        
        Args:
            symbol: The trading symbol (e.g., 'BTC')
            lookback_days: Number of days to look back (default: 1)
            timespan: Time interval for the data (default: 'minute')
            timezone: Timezone for timestamp conversion (default: 'America/New_York')
            
        Returns:
            A pandas DataFrame with merged recent data
        """
        from_date = datetime.now() - timedelta(days=lookback_days)
        return self.get_merged_data(
            symbol=symbol,
            from_date=from_date,
            timespan=timespan,
            timezone=timezone
        )


# Main function to test the DataCollator
def main():
    """
    Test function to demonstrate the usage of DataCollator.
    """
    import os
    import sys
    from pathlib import Path
    
    # Add the project root to the path to import the connectors
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.data_ingestion.polygon_connector import PolygonConnector
    from src.data_ingestion.hyperliquid_connector import HyperLiquidConnector

    SYMBOL = "BTC"
    
    # Initialize connectors
    # You may need to set up API keys as environment variables
    polygon_api_key = os.environ.get("POLYGON_API_KEY")
    if not polygon_api_key:
        print("Warning: POLYGON_API_KEY environment variable not set")
    
    # First, download the Hyperliquid data if needed
    from_date = "20250305"  # Example date format    
    # Check if the file already exists
    polygon_connector = PolygonConnector(tickers=['SPY', 'QQQ', 'VXX'])
    hyperliquid_connector = HyperLiquidConnector(f"./{SYMBOL}")
    # yfinance_connector = YFinanceConnector(tickers="^VIX")
    
    
    # Initialize the DataCollator
    data_collator = DataCollator(
        polygon_connector=polygon_connector,
        hyperliquid_connector=hyperliquid_connector
    )
    
    try:
        # Convert the date format for testing
        from_date_dt = datetime.strptime(from_date, "%Y%m%d")
        to_date_dt = from_date_dt + timedelta(days=1)
        
        merged_data = data_collator.get_merged_data(
            symbol=SYMBOL,
            from_date=from_date_dt,
            to_date=to_date_dt,
            timespan="minute"
        )
        
        print(f"Successfully fetched and merged data!")
        print(f"Shape of merged data: {merged_data.shape}")
        print("\nFirst few rows of merged data:")
        print(merged_data.head())
        
        # Save to CSV for inspection
        output_file = f"./data/{SYMBOL}_merged_data_from_{from_date_dt.date()}.csv"
        merged_data.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        # Display some basic statistics
        print("\nBasic statistics:")
        print(f"Time range: {merged_data['timestamp'].min()} to {merged_data['timestamp'].max()}")
        
        # Check for missing data
        missing_count = merged_data.isna().sum()
        print("\nMissing data count per column:")
        print(missing_count[missing_count > 0])
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


