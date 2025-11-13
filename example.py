"""
Simple example demonstrating the Heston model and arbitrage detection.
"""

import numpy as np
from heston_model import HestonModel
from wrds_connector import generate_sample_data
from arbitrage_detector import ArbitrageDetector


def example_heston_pricing():
    """Example: Price a call option using Heston model."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Heston Model Option Pricing")
    print("="*70)
    
    # Set up parameters
    S0 = 100      # Current stock price
    K = 100       # Strike price
    r = 0.05      # Risk-free rate (5%)
    T = 1.0       # Time to maturity (1 year)
    v0 = 0.04     # Initial variance
    kappa = 2.0   # Mean reversion rate
    theta = 0.04  # Long-term variance
    sigma = 0.3   # Volatility of volatility
    rho = -0.5    # Correlation
    
    print(f"\nParameters:")
    print(f"  Stock Price (S0): ${S0}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Time to Maturity (T): {T} years")
    print(f"  Risk-free Rate (r): {r*100}%")
    print(f"  Initial Variance (v0): {v0}")
    print(f"  Mean Reversion (kappa): {kappa}")
    print(f"  Long-term Variance (theta): {theta}")
    print(f"  Vol of Vol (sigma): {sigma}")
    print(f"  Correlation (rho): {rho}")
    
    # Create model
    model = HestonModel(S0, K, r, T, v0, kappa, theta, sigma, rho)
    
    # Price options
    call_price = model.call_price()
    put_price = model.put_price()
    
    print(f"\nResults:")
    print(f"  Call Price: ${call_price:.4f}")
    print(f"  Put Price: ${put_price:.4f}")
    
    # Verify put-call parity
    parity_check = call_price - put_price - (S0 - K * np.exp(-r * T))
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P - (S - Ke^(-rT)) = {parity_check:.6f}")
    print(f"  (Should be close to 0)")


def example_arbitrage_detection():
    """Example: Detect arbitrage opportunities."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Arbitrage Detection")
    print("="*70)
    
    # Generate sample data
    print("\nGenerating sample options data...")
    options_df, stock_df = generate_sample_data(ticker='AAPL', days=10)
    stock_price = stock_df['stock_price'].iloc[-1]
    
    print(f"Stock Price: ${stock_price:.2f}")
    print(f"Total Options: {len(options_df)}")
    
    # Get latest options
    latest_date = options_df['date'].max()
    latest_options = options_df[options_df['date'] == latest_date]
    
    # Create detector
    detector = ArbitrageDetector(
        latest_options,
        stock_price,
        risk_free_rate=0.05,
        tolerance=0.01
    )
    
    # Detect arbitrage
    print(f"\nAnalyzing {len(latest_options)} options for arbitrage...")
    
    # Put-Call Parity
    pcp = detector.detect_put_call_parity_violations()
    print(f"\nPut-Call Parity Violations: {len(pcp)}")
    if len(pcp) > 0:
        print(f"  Total Expected Profit: ${pcp['expected_profit'].sum():.2f}")
        print("\n  Top Opportunity:")
        best = pcp.iloc[0]
        print(f"    Strike: ${best['strike']:.2f}")
        print(f"    Expiry: {best['T']:.3f} years")
        print(f"    Strategy: {best['strategy']}")
        print(f"    Expected Profit: ${best['expected_profit']:.2f}")
    
    # Butterfly Spreads
    butterfly = detector.detect_butterfly_arbitrage()
    print(f"\nButterfly Arbitrage: {len(butterfly)}")
    if len(butterfly) > 0:
        print(f"  Total Expected Profit: ${butterfly['expected_profit'].sum():.2f}")
    
    # Calendar Spreads
    calendar = detector.detect_calendar_arbitrage()
    print(f"\nCalendar Arbitrage: {len(calendar)}")
    if len(calendar) > 0:
        print(f"  Total Expected Profit: ${calendar['expected_profit'].sum():.2f}")
    
    # Box Spreads
    box = detector.detect_box_spread_arbitrage()
    print(f"\nBox Spread Arbitrage: {len(box)}")
    if len(box) > 0:
        print(f"  Total Expected Profit: ${box['expected_profit'].sum():.2f}")


def example_model_calibration():
    """Example: Calibrate Heston model to market prices."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Model Calibration")
    print("="*70)
    
    # Generate sample data
    options_df, stock_df = generate_sample_data(ticker='MSFT')
    stock_price = stock_df['stock_price'].iloc[-1]
    
    print(f"\nStock Price: ${stock_price:.2f}")
    
    # Select options for calibration
    calls = options_df[
        (options_df['cp_flag'] == 'C') &
        (options_df['T'] > 0.08) &
        (options_df['T'] < 0.3)
    ].head(10)
    
    print(f"Using {len(calls)} call options for calibration")
    
    # Extract data
    market_prices = calls['mid_price'].values
    K_list = calls['strike_price'].values
    T_list = calls['T'].values
    r = 0.05
    
    print("\nCalibrating Heston model...")
    params = HestonModel.calibrate(
        market_prices, stock_price, K_list, r, T_list, K_list, option_type='call'
    )
    
    print("\nCalibrated Parameters:")
    print(f"  v0 (initial variance): {params['v0']:.6f}")
    print(f"  kappa (mean reversion): {params['kappa']:.6f}")
    print(f"  theta (long-term variance): {params['theta']:.6f}")
    print(f"  sigma (vol of vol): {params['sigma']:.6f}")
    print(f"  rho (correlation): {params['rho']:.6f}")
    
    # Calculate pricing errors
    print("\nPricing Errors:")
    total_error = 0
    for i, row in calls.iterrows():
        model = HestonModel(
            stock_price, row['strike_price'], r, row['T'],
            params['v0'], params['kappa'], params['theta'], 
            params['sigma'], params['rho']
        )
        heston_price = model.call_price()
        market_price = row['mid_price']
        error = abs(heston_price - market_price)
        total_error += error
        
        print(f"  K={row['strike_price']:.2f}, T={row['T']:.3f}: "
              f"Market=${market_price:.2f}, Heston=${heston_price:.2f}, "
              f"Error=${error:.2f}")
    
    print(f"\nMean Absolute Error: ${total_error/len(calls):.4f}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HESTON MODEL AND ARBITRAGE DETECTION EXAMPLES")
    print("="*70)
    
    # Run examples
    example_heston_pricing()
    example_arbitrage_detection()
    example_model_calibration()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")
