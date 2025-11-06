import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
from pathlib import Path
from typing import Optional, Tuple

def load_options_data(csv_path) -> Optional[Tuple[pd.DataFrame, str]]:
    """Load and prepare options data."""
    df = pd.read_csv(csv_path)
    
    # Find date column
    date_col = None
    for col in ['date', 'quote_date', 'datadate']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"Error: No date column found. Available columns: {df.columns.tolist()}")
        return None
    
    df['date'] = pd.to_datetime(df[date_col])
    return df, date_col

def get_user_selection(df):
    """Interactive menu for filtering options."""
    print("\n" + "="*60)
    print("OPTION DATA VISUALIZATION")
    print("="*60)
    
    # Show available options
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nAvailable Call/Put flags: {sorted(df['cp_flag'].unique())}")
    
    # Get option type
    cp_flag = input("\nEnter option type (C for Call, P for Put): ").strip().upper()
    if cp_flag not in df['cp_flag'].str.upper().unique():
        print(f"Invalid option type. Using 'C'")
        cp_flag = 'C'
    
    # Show available strikes
    strikes = sorted(df['strike_price'].unique())
    print(f"\nAvailable strikes: {strikes[:10]}... ({len(strikes)} total)")
    print(f"Range: ${strikes[0]} to ${strikes[-1]}")
    
    try:
        strike = float(input("Enter strike price (or press Enter for ATM): ").strip() or df['strike_price'].quantile(0.5))
        # Find closest available strike
        strike = min(strikes, key=lambda x: abs(x - strike))
    except ValueError:
        strike = df['strike_price'].quantile(0.5)
    
    # Show available expiries
    if 'expiration' in df.columns:
        expiries = sorted(df['expiration'].unique())
        print(f"\nAvailable expiration dates: {expiries[:5]}... ({len(expiries)} total)")
        expiry = input("Enter expiration date (YYYY-MM-DD) or press Enter for first: ").strip() or expiries[0]
    else:
        expiry = None
    
    # Time range selection
    print("\n" + "="*60)
    print("TIME RANGE OPTIONS:")
    print("="*60)
    print("1. Daily (1 month range)")
    print("2. Weekly (3 months range)")
    print("3. Monthly (1 year range)")
    print("4. All data available")
    
    time_choice = input("\nSelect time range (1-4): ").strip() or "1"
    
    return cp_flag, strike, expiry, time_choice

def filter_data(df, cp_flag, strike, expiry, time_choice):
    """Filter data based on user selections."""
    # Build query
    query_parts = [f"cp_flag == '{cp_flag}'", f"strike_price == {strike}"]
    if expiry:
        query_parts.append(f"expiration == '{expiry}'")
    
    inst = df.query(" and ".join(query_parts)).sort_values("date").copy()
    
    if len(inst) == 0:
        print("Error: No data found matching the filter criteria.")
        return None, None, None
    
    # Apply time range filter
    max_date = inst['date'].max()
    if time_choice == "1":
        min_date = max_date - pd.Timedelta(days=30)
        resample_freq = "D"
        freq_name = "Daily"
    elif time_choice == "2":
        min_date = max_date - pd.Timedelta(days=90)
        resample_freq = "W"
        freq_name = "Weekly"
    elif time_choice == "3":
        min_date = max_date - pd.Timedelta(days=365)
        resample_freq = "M"
        freq_name = "Monthly"
    else:
        min_date = inst['date'].min()
        resample_freq = "D"
        freq_name = "All Data (Daily)"
    
    inst = inst[inst['date'] >= min_date]
    
    print(f"\nFiltered to {len(inst)} data points ({freq_name})")
    print(f"Date range: {inst['date'].min().date()} to {inst['date'].max().date()}")
    
    return inst, resample_freq, freq_name

def create_ohlc(inst, resample_freq):
    """Create OHLC data from mid prices."""
    inst_indexed = inst.set_index('date')
    
    ohlc = pd.DataFrame({
        "Open":  inst_indexed["mid"].resample(resample_freq).first(),
        "High":  inst_indexed["mid"].resample(resample_freq).max(),
        "Low":   inst_indexed["mid"].resample(resample_freq).min(),
        "Close": inst_indexed["mid"].resample(resample_freq).last(),
    }).dropna()
    
    # Use implied volatility as volume proxy (scaled for visibility)
    if "impl_volatility" in inst_indexed.columns:
        ohlc["Volume"] = (inst_indexed["impl_volatility"].resample(resample_freq).mean() * 1000).fillna(0)
    else:
        ohlc["Volume"] = 100  # Default value
    
    return ohlc

def plot_price_chart(inst, cp_flag, strike, expiry, save_dir):
    """Plot mid price with bid-ask spread."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    inst_plot = inst.set_index('date')
    inst_plot[["mid"]].plot(ax=ax, legend=False, linewidth=2, color='blue')
    ax.fill_between(inst_plot.index, inst_plot["best_bid"], inst_plot["best_offer"], 
                    alpha=0.3, color='lightblue', label='Bid-Ask Spread')
    
    ax.set_title(f"Option Price - {cp_flag} ${strike} (Exp: {expiry})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Option Price ($)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(['Mid Price', 'Bid-Ask Band'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / "option_price_bidask.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

def plot_candlestick(ohlc, resample_freq, freq_name, cp_flag, strike, expiry, save_dir):
    """Plot candlestick chart."""
    # Determine moving averages based on frequency
    if resample_freq == "D":
        mav = (5, 20)  # 5-day and 20-day
    elif resample_freq == "W":
        mav = (4, 12)  # 4-week and 12-week
    else:  # Monthly
        mav = (3, 6)   # 3-month and 6-month
    
    # Create custom style
    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        marketcolors=mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit"),
        gridstyle="-", gridcolor="0.85"
    )
    
    # Plot
    save_path = save_dir / f"candles_{resample_freq}.png"
    mpf.plot(
        ohlc, type="candle", volume=True,
        mav=mav,
        figratio=(16, 9), figscale=1.2, style=style,
        title=f"{freq_name} Candles - {cp_flag} ${strike} Option (Exp: {expiry})", 
        ylabel="Option Price ($)", 
        ylabel_lower="Implied Vol (scaled)",
        tight_layout=True, 
        savefig=str(save_path)
    )
    
    print(f"Saved: {save_path}")

def main():
    # Get save directory (Visualisation folder)
    save_dir = Path(__file__).parent
    
    # Load data
    csv_path = save_dir.parent / "aapl_options_jan2023.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    result = load_options_data(csv_path)
    if result is None:
        return
    
    df, date_col = result
    
    # Get user selections
    cp_flag, strike, expiry, time_choice = get_user_selection(df)
    
    # Filter data
    inst, resample_freq, freq_name = filter_data(df, cp_flag, strike, expiry, time_choice)
    if inst is None:
        return
    
    # Create OHLC
    ohlc = create_ohlc(inst, resample_freq)
    print(f"Generated {len(ohlc)} {freq_name} candles")
    
    # Generate visualizations
    print("\nGenerating charts...")
    plot_price_chart(inst, cp_flag, strike, expiry, save_dir)
    plot_candlestick(ohlc, resample_freq, freq_name, cp_flag, strike, expiry, save_dir)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print(f"Charts saved to: {save_dir}")
    print("="*60)

if __name__ == "__main__":
    main()