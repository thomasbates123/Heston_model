# Heston calibration (price-based) — full script
import numpy as np
import pandas as pd
from scipy import integrate, optimize
from math import log, exp, sqrt, pi

# ---------------------------
# Heston characteristic function and integrals
# ---------------------------

def heston_cf(phi, params, S, K, T, r, q, j):
    """
    Characteristic function for Heston model.
    phi : integration variable (can be complex)
    params: tuple/list (kappa, theta, sigma_v, rho, v0)
    S, K, T, r, q: scalars
    j: 1 or 2 (probability index)
    returns: complex
    """
    kappa, theta, sigma_v, rho, v0 = params
    # complex i
    i = 1j
    # parameters depending on j
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    # d and g per Heston
    alpha = - (phi**2) / 2.0 - i * u * phi
    beta = kappa - rho * sigma_v * i * phi
    gamma = (sigma_v**2) / 2.0

    d = np.sqrt(beta**2 - 4.0 * alpha * gamma)
    # Avoid sign ambiguity: choose branch with positive real part for stability
    # compute g
    g = (beta - d) / (beta + d)

    # terms for exponentials
    # C(t) and D(t) per usual closed-form
    exp_dt = np.exp(-d * T)
    # Avoid division by zero issues
    denom = 1.0 - g * exp_dt
    C = (r - q) * i * phi * T + (a / (sigma_v**2)) * ((beta - d) * T - 2.0 * np.log(denom / (1.0 - g)))
    D = ((beta - d) / (sigma_v**2)) * ((1.0 - exp_dt) / denom)

    # characteristic function
    cf = np.exp(C + D * v0 + i * phi * np.log(S * np.exp(-q * T)))
    return cf

def integrand_P(phi, params, S, K, T, r, q, j):
    i = 1j
    cf_val = heston_cf(phi - i*0, params, S, K, T, r, q, j)  # phi real
    numerator = np.exp(-1j * phi * np.log(K)) * cf_val
    denom = 1j * phi
    return (numerator / denom).real

def Pj(params, S, K, T, r, q, j):
    """
    Compute P1 or P2 using numerical integration of real part:
    Pj = 1/2 + (1/pi) * integral_0^inf Re( e^{-i phi ln K} * phi^{-1} * cf(phi) ) d phi
    We integrate from 0 to upper and use quad.
    """
    integrand = lambda phi: integrand_P(phi, params, S, K, T, r, q, j)
    # integration upper bound: 200 or less; you can tune
    upper = 200.0
    val, err = integrate.quad(integrand, 0.0, upper, limit=200, epsabs=1e-6, epsrel=1e-6)
    return 0.5 + val / pi

def heston_price_call(params, S, K, T, r, q):
    """
    Compute European call price under Heston using probabilities P1 and P2.
    """
    P1 = Pj(params, S, K, T, r, q, j=1)
    P2 = Pj(params, S, K, T, r, q, j=2)
    discounted_forward = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    # Ensure non-negative
    return max(discounted_forward, 0.0)

# For puts, use put-call parity
def heston_price(params, S, K, T, r, q, cp_flag):
    call = heston_price_call(params, S, K, T, r, q)
    if cp_flag.upper() in ['C', 'CALL']:
        return call
    else:
        # P = C - S e^{-qT} + K e^{-rT}
        return call - S * np.exp(-q * T) + K * np.exp(-r * T)

# ---------------------------
# Objective & calibration
# ---------------------------

def calibration_objective(x, df_slice, r, q, weight_by='vega'):
    """
    x: parameter vector [kappa, theta, sigma_v, rho, v0]
    df_slice: pandas DataFrame with S, strike_price, mid, T, cp_flag
    r, q: rates
    weight_by: None or 'vega' or 'mid' for weighting
    """
    # enforce positivity inside objective by large penalty if violated
    kappa, theta, sigma_v, rho, v0 = x
    if (kappa <= 0) or (theta <= 0) or (sigma_v <= 0) or (v0 <= 0) or (abs(rho) >= 1):
        return 1e10

    params = (kappa, theta, sigma_v, rho, v0)
    errs = []
    for _, row in df_slice.iterrows():
        S = float(row['S'])
        K = float(row['strike_price'])
        T = float(row['T'])
        cp_flag = row['cp_flag']
        market_price = float(row['mid'])
        if T <= 0 or market_price <= 0:
            continue
        model_price = heston_price(params, S, K, T, r, q, cp_flag)
        # weighting
        if weight_by == 'mid':
            w = 1.0 / max(1e-4, market_price)
        else:
            w = 1.0
        errs.append(w * (model_price - market_price)**2)
    if len(errs) == 0:
        return 1e8
    return float(np.sum(errs) / len(errs))

# ---------------------------
# Example usage
# ---------------------------

def prepare_dataframe_for_calibration(df, quote_date, r, q, min_mid=0.01, max_moneyness=0.5):
    """
    Filter and prepare the df for a single calibration date:
    df: merged dataframe of options + underlying
    quote_date: date to calibrate on (string or pd.Timestamp)
    r, q: scalars (annual continuous rates)
    returns: filtered df_slice with columns S, strike_price, mid, T, cp_flag
    """
    d = pd.to_datetime(quote_date)
    df_date = df[df['date'] == d].copy()
    # compute T in years
    df_date['T'] = (pd.to_datetime(df_date['exdate']) - pd.to_datetime(df_date['date'])).dt.days / 365.0
    # basic filters: positive mid, positive T, moneyness limit
    df_date = df_date[(df_date['mid'] > min_mid) & (df_date['T'] > 0)]
    S = float(df_date['S'].iloc[0])
    df_date['moneyness'] = np.abs(np.log(df_date['strike_price'] / S))
    df_date = df_date[df_date['moneyness'] <= max_moneyness]
    # optional: choose only calls or calls+puts - calls are typical
    return df_date

# ---------------------------
# Running calibration
# ---------------------------

def run_calibration(df, quote_date, r=0.01, q=0.0):
    # Prepare data
    df_slice = prepare_dataframe_for_calibration(df, quote_date, r, q)
    print(f"Using {len(df_slice)} options for calibration on {quote_date}")

    # initial guess: (kappa, theta, sigma_v, rho, v0)
    x0 = np.array([1.0, 0.04, 0.5, -0.5, 0.04])  # typical starting values
    # bounds: kappa>0, theta>0, sigma_v>0, rho in (-0.999,0.999), v0>0
    bnds = [(1e-4, 10.0), (1e-6, 2.0), (1e-4, 3.0), (-0.999, 0.999), (1e-6, 2.0)]

    res = optimize.minimize(
        calibration_objective,
        x0,
        args=(df_slice, r, q, 'mid'),
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': 200}
    )
    print("Optimize success:", res.success, res.message)
    print("Parameters:", res.x)
    return res, df_slice

# Example: call run_calibration assuming you have df available
# res, used_df = run_calibration(df, quote_date='2023-01-03', r=0.02, q=0.0)

# ---------------------------
# 📊 USING REAL AAPL DATA
# ---------------------------

def load_aapl_data():
    """
    Load the real AAPL options and underlying data from CSV files.
    """
    print("📂 Loading AAPL data from CSV files...")
    
    # Load options data
    options_df = pd.read_csv('aapl_options_data.csv')
    print(f"   Options data: {len(options_df)} records")
    
    # Load underlying prices
    underlying_df = pd.read_csv('aapl_underlying_prices.csv')
    print(f"   Underlying data: {len(underlying_df)} records")
    
    # Convert date columns
    options_df['date'] = pd.to_datetime(options_df['date'])
    options_df['exdate'] = pd.to_datetime(options_df['exdate'])
    underlying_df['date'] = pd.to_datetime(underlying_df['date'])
    
    # Rename underlying price column to match expected format
    underlying_df = underlying_df.rename(columns={'close': 'S'})
    
    # Merge options with underlying prices
    merged_df = options_df.merge(underlying_df, on='date', how='left')
    
    print(f"   Merged data: {len(merged_df)} records")
    print(f"   Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"   Available dates: {merged_df['date'].nunique()} unique dates")
    
    return merged_df

def run_aapl_calibration(calibration_date='2023-01-03'):
    """
    Run Heston calibration using real AAPL options data.
    """
    print("🍎 AAPL HESTON CALIBRATION WITH REAL DATA")
    print("=" * 60)
    
    # Load the data
    df = load_aapl_data()
    
    # Check available dates
    available_dates = sorted(df['date'].unique())
    print(f"\nAvailable calibration dates:")
    for i, date in enumerate(available_dates[:10]):  # Show first 10
        date_str = date.strftime('%Y-%m-%d')
        count = len(df[df['date'] == date])
        print(f"   {date_str}: {count} options")
    if len(available_dates) > 10:
        print(f"   ... and {len(available_dates) - 10} more dates")
    
    # Set calibration date
    cal_date = pd.to_datetime(calibration_date)
    if cal_date not in available_dates:
        print(f"\n⚠️  Date {calibration_date} not available. Using first available date: {available_dates[0].strftime('%Y-%m-%d')}")
        cal_date = available_dates[0]
        calibration_date = cal_date.strftime('%Y-%m-%d')
    
    # Market parameters
    r = 0.02  # 2% risk-free rate
    q = 0.0   # 0% dividend yield for AAPL (approximate)
    
    print(f"\n📅 Calibrating for date: {calibration_date}")
    print(f"📊 Market parameters: r={r:.1%}, q={q:.1%}")
    
    # Show data summary for calibration date
    day_data = df[df['date'] == cal_date]
    print(f"\n📈 Data summary for {calibration_date}:")
    print(f"   • Total options: {len(day_data)}")
    print(f"   • Underlying price (S): ${day_data['S'].iloc[0]:.2f}")
    print(f"   • Call options: {len(day_data[day_data['cp_flag'] == 'C'])}")
    print(f"   • Put options: {len(day_data[day_data['cp_flag'] == 'P'])}")
    print(f"   • Strike range: ${day_data['strike_price'].min():.0f} - ${day_data['strike_price'].max():.0f}")
    print(f"   • Time to expiry range: {day_data['T'].min():.3f} - {day_data['T'].max():.3f} years")
    
    # Run calibration
    try:
        result, used_data = run_calibration(df, calibration_date, r=r, q=q)
        
        if result.success:
            print(f"\n✅ CALIBRATION SUCCESSFUL!")
            
            # Extract and display parameters
            kappa, theta, sigma_v, rho, v0 = result.x
            print(f"\n🎯 CALIBRATED HESTON PARAMETERS:")
            print(f"   • κ (mean reversion speed):  {kappa:.4f}")
            print(f"   • θ (long-term variance):    {theta:.4f} (vol = {np.sqrt(theta):.1%})")
            print(f"   • σᵥ (volatility of vol):    {sigma_v:.4f}")
            print(f"   • ρ (correlation):           {rho:.4f}")
            print(f"   • v₀ (initial variance):     {v0:.4f} (vol = {np.sqrt(v0):.1%})")
            print(f"   • Final RMSE:                {np.sqrt(result.fun):.6f}")
            
            # Test model performance on a few options
            print(f"\n🧪 MODEL PERFORMANCE TEST:")
            test_options = used_data.head(5)  # Test first 5 options
            
            for i, (_, option) in enumerate(test_options.iterrows()):
                S = option['S']
                K = option['strike_price']
                T = option['T']
                cp_flag = option['cp_flag']
                market_price = option['mid']
                
                model_price = heston_price(result.x, S, K, T, r, q, cp_flag)
                error = abs(model_price - market_price)
                error_pct = (error / market_price) * 100
                
                print(f"   {i+1}. {cp_flag} K=${K:.0f} T={T:.3f}y: Market=${market_price:.3f} Model=${model_price:.3f} Error={error_pct:.1f}%")
            
            return result, used_data, df
            
        else:
            print(f"\n❌ CALIBRATION FAILED:")
            print(f"   Message: {result.message}")
            return None, None, df
            
    except Exception as e:
        print(f"\n💥 ERROR during calibration: {str(e)}")
        return None, None, df

def analyze_multiple_dates():
    """
    Run calibration for multiple dates to see parameter evolution.
    """
    print("\n📈 MULTI-DATE CALIBRATION ANALYSIS")
    print("=" * 50)
    
    df = load_aapl_data()
    available_dates = sorted(df['date'].unique())
    
    # Select a few dates for analysis (e.g., every 5th date)
    analysis_dates = available_dates[::5][:5]  # Every 5th date, max 5 dates
    
    results = []
    for date in analysis_dates:
        date_str = date.strftime('%Y-%m-%d')
        print(f"\n🔄 Calibrating for {date_str}...")
        
        try:
            result, used_data = run_calibration(df, date_str, r=0.02, q=0.0)
            if result.success:
                kappa, theta, sigma_v, rho, v0 = result.x
                results.append({
                    'date': date_str,
                    'kappa': kappa,
                    'theta': theta,
                    'sigma_v': sigma_v,
                    'rho': rho,
                    'v0': v0,
                    'rmse': np.sqrt(result.fun),
                    'n_options': len(used_data)
                })
                print(f"   ✅ Success: κ={kappa:.3f}, θ={theta:.4f}, σᵥ={sigma_v:.3f}, ρ={rho:.3f}, v₀={v0:.4f}")
            else:
                print(f"   ❌ Failed: {result.message}")
        except Exception as e:
            print(f"   💥 Error: {str(e)}")
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n📊 PARAMETER EVOLUTION SUMMARY:")
        print(results_df.round(4))
        
        # Save results
        results_df.to_csv('aapl_heston_calibration_results.csv', index=False)
        print(f"\n💾 Results saved to 'aapl_heston_calibration_results.csv'")
        
        return results_df
    
    return None

# ---------------------------
# 🎯 MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    print("🚀 HESTON MODEL CALIBRATION")
    print("Choose an option:")
    print("1. Run single-date AAPL calibration")
    print("2. Run multi-date analysis")
    print("3. Run example with sample data")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Single date calibration with real AAPL data
        date = input("Enter calibration date (YYYY-MM-DD) or press Enter for 2023-01-03: ").strip()
        if not date:
            date = "2023-01-03"
        
        result, used_data, df = run_aapl_calibration(date)
        
    elif choice == "2":
        # Multi-date analysis
        results_df = analyze_multiple_dates()
        
    elif choice == "3":
        # Example with sample data  
        print("\n📋 This would run with sample data (functions not included in this version)")
        print("📊 Running AAPL calibration instead...")
        result, used_data, df = run_aapl_calibration("2023-01-03")
        
    else:
        print("Invalid choice. Running default single-date calibration...")
        result, used_data, df = run_aapl_calibration("2023-01-03")
