"""
WRDS Data Connector for Options Data

This module provides functionality to connect to WRDS and fetch options data
for arbitrage analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class WRDSConnector:
    """
    Connector for fetching options data from WRDS (Wharton Research Data Services).
    
    Note: Requires WRDS account credentials. See WRDS documentation for setup:
    https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/
    """
    
    def __init__(self, username=None):
        """
        Initialize WRDS connection.
        
        Parameters:
        -----------
        username : str, optional
            WRDS username. If None, will use credentials from .pgpass file
        """
        self.username = username
        self.db = None
        
    def connect(self):
        """
        Establish connection to WRDS database.
        
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        try:
            import wrds
            self.db = wrds.Connection(wrds_username=self.username)
            return True
        except Exception as e:
            print(f"Failed to connect to WRDS: {e}")
            print("Make sure you have WRDS credentials configured.")
            return False
    
    def fetch_option_data(self, ticker, start_date, end_date, library='optionm'):
        """
        Fetch options data for a given ticker and date range.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        library : str
            WRDS library to use ('optionm' for OptionMetrics)
            
        Returns:
        --------
        pd.DataFrame
            Options data with columns including strike, maturity, price, volume, etc.
        """
        if self.db is None:
            if not self.connect():
                return None
        
        try:
            # Query for option prices from OptionMetrics
            query = f"""
                SELECT 
                    date,
                    exdate,
                    cp_flag,
                    strike_price,
                    best_bid,
                    best_offer,
                    volume,
                    open_interest,
                    impl_volatility
                FROM {library}.opprcd{start_date[:4]}
                WHERE secid = (
                    SELECT secid 
                    FROM {library}.securd 
                    WHERE ticker = '{ticker}'
                    LIMIT 1
                )
                AND date >= '{start_date}'
                AND date <= '{end_date}'
                ORDER BY date, exdate, strike_price
            """
            
            df = self.db.raw_sql(query)
            
            # Calculate mid price
            df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
            
            # Calculate time to maturity in years
            df['date'] = pd.to_datetime(df['date'])
            df['exdate'] = pd.to_datetime(df['exdate'])
            df['T'] = (df['exdate'] - df['date']).dt.days / 365.0
            
            return df
            
        except Exception as e:
            print(f"Error fetching option data: {e}")
            return None
    
    def fetch_underlying_price(self, ticker, start_date, end_date, library='optionm'):
        """
        Fetch underlying stock prices.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        library : str
            WRDS library to use
            
        Returns:
        --------
        pd.DataFrame
            Stock prices with date and close price
        """
        if self.db is None:
            if not self.connect():
                return None
        
        try:
            query = f"""
                SELECT 
                    date,
                    close as stock_price
                FROM {library}.secprd
                WHERE secid = (
                    SELECT secid 
                    FROM {library}.securd 
                    WHERE ticker = '{ticker}'
                    LIMIT 1
                )
                AND date >= '{start_date}'
                AND date <= '{end_date}'
                ORDER BY date
            """
            
            df = self.db.raw_sql(query)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            print(f"Error fetching underlying price: {e}")
            return None
    
    def close(self):
        """Close WRDS connection."""
        if self.db is not None:
            self.db.close()


def generate_sample_data(ticker='AAPL', days=30):
    """
    Generate sample options data for testing when WRDS is not available.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    days : int
        Number of days of data to generate
        
    Returns:
    --------
    tuple
        (options_df, stock_price_df)
    """
    np.random.seed(42)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Sample stock price around $150
    stock_price = 150 + np.random.randn() * 5
    
    # Generate options data
    options_data = []
    strikes = np.arange(140, 161, 2.5)
    maturities = [30, 60, 90]  # days to expiration
    
    for date in dates[-5:]:  # Last 5 days
        for maturity_days in maturities:
            expiry = date + timedelta(days=maturity_days)
            T = maturity_days / 365.0
            
            for strike in strikes:
                # Simple Black-Scholes approximation for realistic prices
                moneyness = stock_price / strike
                r = 0.05
                sigma = 0.25
                
                d1 = (np.log(moneyness) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                
                from scipy.stats import norm
                
                call_price = stock_price * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
                put_price = strike * np.exp(-r * T) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
                
                # Add some noise
                call_bid = max(0.01, call_price * (1 - 0.02 * np.random.rand()))
                call_offer = call_price * (1 + 0.02 * np.random.rand())
                put_bid = max(0.01, put_price * (1 - 0.02 * np.random.rand()))
                put_offer = put_price * (1 + 0.02 * np.random.rand())
                
                # Call option
                options_data.append({
                    'date': date,
                    'exdate': expiry,
                    'cp_flag': 'C',
                    'strike_price': strike,
                    'best_bid': call_bid,
                    'best_offer': call_offer,
                    'volume': np.random.randint(10, 1000),
                    'open_interest': np.random.randint(100, 10000),
                    'impl_volatility': sigma + np.random.randn() * 0.02,
                    'T': T,
                    'mid_price': (call_bid + call_offer) / 2
                })
                
                # Put option
                options_data.append({
                    'date': date,
                    'exdate': expiry,
                    'cp_flag': 'P',
                    'strike_price': strike,
                    'best_bid': put_bid,
                    'best_offer': put_offer,
                    'volume': np.random.randint(10, 1000),
                    'open_interest': np.random.randint(100, 10000),
                    'impl_volatility': sigma + np.random.randn() * 0.02,
                    'T': T,
                    'mid_price': (put_bid + put_offer) / 2
                })
    
    options_df = pd.DataFrame(options_data)
    
    # Stock price data
    stock_data = []
    for date in dates:
        stock_data.append({
            'date': date,
            'stock_price': stock_price + np.random.randn() * 2
        })
    
    stock_df = pd.DataFrame(stock_data)
    
    return options_df, stock_df
