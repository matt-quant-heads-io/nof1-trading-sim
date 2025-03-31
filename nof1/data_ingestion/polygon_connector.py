import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

class PolygonConnector:
    """
    A connector for fetching OHLC data from Polygon.io API.
    """
    
    def __init__(self, tickers=None):
        """
        Initialize the Polygon connector.
        
        Args:
            tickers (list, optional): List of ticker symbols to fetch data for.
                                     Defaults to ['QQQ', 'SPY'].
        """
        load_dotenv()
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment variables")
        
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        self.tickers = tickers if tickers else ['QQQ', 'SPY']
    
    def fetch_ohlc(self, timespan='day', multiplier=1, from_date=None, to_date=None, limit=1000):
        """
        Fetch OHLC data for the specified tickers and time range.
        
        Args:
            timespan (str): The timespan unit. Options: minute, hour, day, week, month, quarter, year
            multiplier (int): The multiplier for the timespan
            from_date (str or datetime): Start date in 'YYYY-MM-DD' format or datetime object
            to_date (str or datetime): End date in 'YYYY-MM-DD' format or datetime object
            limit (int): Maximum number of results to return
            
        Returns:
            pandas.DataFrame: DataFrame with OHLC data for all tickers, columns prefixed with ticker
        """
        # Set default dates if not provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        elif isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
            
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        all_dfs = []
        
        for ticker in self.tickers:
            # Construct the URL
            url = f"{self.base_url}/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": limit,
                "apiKey": self.api_key
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()  # Raise exception for HTTP errors
                data = response.json()
                
                if data['status'] != 'OK':
                    print(f"Error fetching data for {ticker}: {data.get('error', 'Unknown error')}")
                    continue
                
                # Convert results to DataFrame
                if not data.get('results'):
                    print(f"No results found for {ticker}")
                    continue
                    
                df = pd.DataFrame(data['results'])
                
                # Rename columns to prefix with ticker
                df = df.rename(columns={
                    'o': f'{ticker}_o',
                    'h': f'{ticker}_h',
                    'l': f'{ticker}_l',
                    'c': f'{ticker}_c',
                    'v': f'{ticker}_v',
                    'vw': f'{ticker}_vw',
                    'n': f'{ticker}_n'
                })
                
                # Keep 't' column for merging
                all_dfs.append(df)
                
                # Respect rate limits
                time.sleep(0.2)  # 200ms delay between requests
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Merge all dataframes on timestamp 't'
        result_df = all_dfs[0]
        for df in all_dfs[1:]:
            result_df = pd.merge(result_df, df, on='t', how='inner')
        
        # Convert timestamp to datetime
        result_df['timestamp'] = pd.to_datetime(result_df['t'], unit='ms')
        
        # Sort by timestamp
        result_df = result_df.sort_values('timestamp') #2025-02-28 00:01:00
        
        return result_df
    
    def fetch_daily_bars(self, days=30):
        """
        Convenience method to fetch daily bars for the last specified number of days.
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: DataFrame with daily OHLC data
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.fetch_ohlc(timespan='day', from_date=from_date)
    
    def fetch_hourly_bars(self, days=7):
        """
        Convenience method to fetch hourly bars for the last specified number of days.
        
        Args:
            days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: DataFrame with hourly OHLC data
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.fetch_ohlc(timespan='hour', from_date=from_date)
    
    def fetch_minute_bars(self, hours=24):
        """
        Convenience method to fetch minute bars for the last specified number of hours.
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            pandas.DataFrame: DataFrame with minute OHLC data
        """
        from_date = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
        return self.fetch_ohlc(timespan='minute', from_date=from_date)
