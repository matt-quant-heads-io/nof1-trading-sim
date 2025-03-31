import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

class YFinanceConnector:
    """
    Connector for fetching data from Yahoo Finance.
    """
    
    def __init__(self, tickers=["SPY"]):
        """
        Initialize the Yahoo Finance connector.
        
        Args:
            tickers (str or list): The ticker symbol(s) to fetch data for. Default is ["SPY"].
                                  Can be a single ticker as string or a list of tickers.
        """
        # Convert single ticker to list for consistent handling
        if isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized YFinanceConnector with tickers: {self.tickers}")
        
    def fetch_data(self, ticker, date_str):
        """
        Fetch 1-minute OHLCV data for a specific date for all tickers.
        
        Args:
            date_str (str): Date in the format 'YYYY-MM-DD'.
            
        Returns:
            pd.DataFrame: DataFrame containing 1-minute OHLCV data for the specified date.
        """
        try:
            date = date_str
            # Parse the input date
            if isinstance(date, str):
                date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Set start and end dates (end date is the next day to include the full day)
            start_date = date
            end_date = date + timedelta(days=1)
            
            # Format dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data from Yahoo Finance
            # tickers_str = " ".join(self.tickers)
            # self.logger.info(f"Fetching 1-minute data for {tickers_str} on {date_str}")
            data = yf.download(
                ticker,
                start=start_str,
                end=end_str,
                interval="1m",
                auto_adjust=True
            )
            # import pdb; pdb.set_trace()
            
            # Check if data is empty
            if data.empty:
                # self.logger.warning(f"No data available for {tickers_str} on {date_str}")
                return pd.DataFrame()
            
            # Reset index to make datetime a column
            data = data.reset_index()
            
            # Handle multi-index columns if present (happens with multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten the multi-index columns
                data.columns = [
                    col[0] if col[0] == 'Datetime' else f"{col[0]}_{col[1].lower()}" 
                    for col in data.columns
                ]
            else:
                # For single ticker, prepend ticker to column names except timestamp
                ticker = self.tickers[0]
                data.columns = [
                    col if col == 'Datetime' else f"{ticker}_{col.lower()}" 
                    for col in data.columns
                ]
            
            # Rename timestamp column
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'timestamp'})
            
            # Filter to include only the requested date
            data['date'] = data['timestamp'].dt.date
            requested_date = date #datetime.strptime(date_str, '%Y-%m-%d').date()
            data = data[data['date'] == requested_date]
            
            # Reformat timestamp to remove timezone information
            data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Drop the temporary date column
            data = data.drop(columns=['date'])
            import pdb; pdb.set_trace()
            
            self.logger.info(f"Successfully fetched {len(data)} records for {date_str}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def set_tickers(self, tickers):
        """
        Change the ticker symbols.
        
        Args:
            tickers (str or list): New ticker symbol(s). Can be a single ticker as string or a list of tickers.
        """
        if isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers
            
        self.logger.info(f"Tickers changed to {self.tickers}")
    
    def add_ticker(self, ticker):
        """
        Add a ticker symbol to the existing list.
        
        Args:
            ticker (str): Ticker symbol to add.
        """
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            self.logger.info(f"Added ticker {ticker}")
        else:
            self.logger.info(f"Ticker {ticker} already exists")
    
    def remove_ticker(self, ticker):
        """
        Remove a ticker symbol from the list.
        
        Args:
            ticker (str): Ticker symbol to remove.
        """
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            self.logger.info(f"Removed ticker {ticker}")
        else:
            self.logger.warning(f"Ticker {ticker} not found in current tickers")

    def merge_data(self, date_str):
        """
        Fetch data for each ticker individually and merge them into a single dataframe.
        This is useful when you want to ensure data for each ticker is fetched separately
        and then combined.
        
        Args:
            date_str (str): Date in the format 'YYYY-MM-DD'.
            
        Returns:
            pd.DataFrame: DataFrame containing merged data from all tickers.
        """
        self.logger.info(f"Fetching and merging data for {len(self.tickers)} tickers on {date_str}")
        
        # Store the original list of tickers
        original_tickers = self.tickers.copy()
        
        merged_df = None
        
        try:
            for ticker in original_tickers:
                # Temporarily set to single ticker
                self.tickers = [ticker]
                
                # Fetch data for this ticker
                ticker_data = self.fetch_data(date_str)
                
                if ticker_data.empty:
                    self.logger.warning(f"No data available for {ticker} on {date_str}")
                    continue
                    
                # For the first ticker, initialize the merged dataframe
                if merged_df is None:
                    merged_df = ticker_data
                else:
                    # Merge with existing data on timestamp
                    merged_df = pd.merge(merged_df, ticker_data, on='timestamp', how='outer')
            
            # Restore original tickers list
            self.tickers = original_tickers
            
            if merged_df is None:
                self.logger.warning(f"No data available for any tickers on {date_str}")
                return pd.DataFrame()
                
            # Sort by timestamp
            merged_df = merged_df.sort_values('timestamp')
            
            self.logger.info(f"Successfully merged data for {len(original_tickers)} tickers with {len(merged_df)} rows")
            return merged_df
            
        except Exception as e:
            # Restore original tickers list in case of error
            self.tickers = original_tickers
            self.logger.error(f"Error merging data: {str(e)}")
            raise
