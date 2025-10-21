"""
Main Script for Options Arbitrage Analysis using Heston Model

This script demonstrates the complete workflow for:
1. Fetching options data from WRDS (or using sample data)
2. Calibrating the Heston model to market prices
3. Detecting arbitrage opportunities
4. Analyzing and reporting results
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

from heston_model import HestonModel
from wrds_connector import WRDSConnector, generate_sample_data
from arbitrage_detector import ArbitrageDetector


def analyze_arbitrage(ticker='AAPL', use_sample_data=True, start_date=None, end_date=None):
    """
    Main function to analyze options arbitrage opportunities.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    use_sample_data : bool
        If True, use generated sample data instead of WRDS
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    """
    print("\n" + "="*70)
    print("OPTIONS ARBITRAGE ANALYSIS USING HESTON MODEL")
    print("="*70)
    print(f"Ticker: {ticker}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Step 1: Fetch or generate data
    print("Step 1: Fetching options data...")
    
    if use_sample_data:
        print("Using sample data for demonstration...")
        options_df, stock_df = generate_sample_data(ticker=ticker)
        stock_price = stock_df['stock_price'].iloc[-1]
        print(f"Generated {len(options_df)} option records")
    else:
        print("Connecting to WRDS...")
        connector = WRDSConnector()
        
        if start_date is None:
            from datetime import timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        options_df = connector.fetch_option_data(ticker, start_date, end_date)
        stock_df = connector.fetch_underlying_price(ticker, start_date, end_date)
        
        if options_df is None or stock_df is None:
            print("Failed to fetch data from WRDS. Using sample data instead.")
            options_df, stock_df = generate_sample_data(ticker=ticker)
        
        stock_price = stock_df['stock_price'].iloc[-1]
        connector.close()
    
    print(f"Current Stock Price: ${stock_price:.2f}")
    print(f"Date Range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"Total Options: {len(options_df)}")
    print(f"  Calls: {len(options_df[options_df['cp_flag'] == 'C'])}")
    print(f"  Puts: {len(options_df[options_df['cp_flag'] == 'P'])}")
    
    # Step 2: Calibrate Heston Model
    print("\nStep 2: Calibrating Heston Model...")
    
    # Select a subset of liquid options for calibration
    calibration_data = options_df[
        (options_df['T'] > 0.08) &  # At least 1 month
        (options_df['T'] < 0.5) &   # Less than 6 months
        (options_df['volume'] > 10)  # Reasonable volume
    ].copy()
    
    if len(calibration_data) > 0:
        # Use calls for calibration
        calls = calibration_data[calibration_data['cp_flag'] == 'C'].head(20)
        
        if len(calls) >= 5:
            try:
                market_prices = calls['mid_price'].values
                K_list = calls['strike_price'].values
                T_list = calls['T'].values
                r = 0.05  # 5% risk-free rate
                
                print(f"Calibrating with {len(calls)} call options...")
                params = HestonModel.calibrate(
                    market_prices, stock_price, K_list, r, T_list, K_list, option_type='call'
                )
                
                print("\nCalibrated Heston Parameters:")
                print(f"  v0 (initial variance): {params['v0']:.6f}")
                print(f"  kappa (mean reversion): {params['kappa']:.6f}")
                print(f"  theta (long-term variance): {params['theta']:.6f}")
                print(f"  sigma (vol of vol): {params['sigma']:.6f}")
                print(f"  rho (correlation): {params['rho']:.6f}")
                
                # Verify calibration with a sample option
                sample_option = calls.iloc[0]
                model = HestonModel(
                    stock_price, 
                    sample_option['strike_price'],
                    r,
                    sample_option['T'],
                    params['v0'],
                    params['kappa'],
                    params['theta'],
                    params['sigma'],
                    params['rho']
                )
                
                heston_price = model.call_price()
                market_price = sample_option['mid_price']
                print(f"\nSample Verification (K={sample_option['strike_price']:.2f}, T={sample_option['T']:.3f}):")
                print(f"  Market Price: ${market_price:.2f}")
                print(f"  Heston Price: ${heston_price:.2f}")
                print(f"  Error: ${abs(heston_price - market_price):.2f}")
                
            except Exception as e:
                print(f"Calibration failed: {e}")
                print("Proceeding with arbitrage detection using market prices...")
        else:
            print("Insufficient data for calibration. Proceeding with market prices...")
    else:
        print("No suitable options found for calibration. Proceeding with market prices...")
    
    # Step 3: Detect Arbitrage Opportunities
    print("\nStep 3: Detecting Arbitrage Opportunities...")
    
    # Use most recent date for analysis
    latest_date = options_df['date'].max()
    latest_options = options_df[options_df['date'] == latest_date].copy()
    
    print(f"Analyzing options as of {latest_date}")
    
    detector = ArbitrageDetector(
        latest_options,
        stock_price,
        risk_free_rate=0.05,
        tolerance=0.01  # $0.01 minimum profit to account for transaction costs
    )
    
    results = detector.detect_all_arbitrage()
    
    # Step 4: Display Detailed Results
    print("\nStep 4: Detailed Arbitrage Opportunities\n")
    
    for arb_type, df in results.items():
        if len(df) > 0:
            print(f"\n{arb_type.replace('_', ' ').upper()}")
            print("-" * 70)
            print(df.to_string(index=False))
            print()
    
    # Step 5: Save Results
    print("\nStep 5: Saving Results...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for arb_type, df in results.items():
        if len(df) > 0:
            filename = f"arbitrage_{arb_type}_{ticker}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {arb_type} results to {filename}")
    
    print("\nAnalysis complete!")
    print("="*70 + "\n")
    
    return results


def main():
    """Command-line interface for the arbitrage analyzer."""
    parser = argparse.ArgumentParser(
        description='Analyze options arbitrage opportunities using the Heston model'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default='AAPL',
        help='Stock ticker symbol (default: AAPL)'
    )
    
    parser.add_argument(
        '--use-wrds',
        action='store_true',
        help='Use WRDS data instead of sample data (requires WRDS credentials)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format'
    )
    
    args = parser.parse_args()
    
    analyze_arbitrage(
        ticker=args.ticker,
        use_sample_data=not args.use_wrds,
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == '__main__':
    main()
