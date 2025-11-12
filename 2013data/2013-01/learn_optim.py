import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# --- Load and prep data ---
df = pd.read_csv('A_df.csv')

# rename and select columns
df = df.rename(columns={'close': 'spot', 'mid': 'mid_price'})
required = ['spot', 'strike', 'T', 'r', 'mid_price','quote_date']
df = df[required].copy()
df['w'] = 1.0

S = df['spot'].iloc[0]       # assume same spot for all rows
Ks = df['strike'].values
Ts = df['T'].values
r = df['r'].iloc[0]
market_prices = df['mid_price'].values

df['date'] = pd.to_datetime(df['quote_date'])
df = df.sort_values('date')

# --- Black-Scholes function ---
def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# --- Objective function ---
def objective(sigma_hat, S, Ks, Ts, r, market_prices):
    sigma = np.exp(sigma_hat)
    model_prices = np.array([bs_call_price(S, K, T, r, sigma) for K, T in zip(Ks, Ts)])
    return np.mean((model_prices - market_prices) ** 2)

# --- Rolling calibration ---
results = []

unique_dates = sorted(df['date'].unique())

for i in range(len(unique_dates) - 1):
    train_dates = unique_dates[:i+1]       # up to and including date i
    test_date = unique_dates[i+1]          # next day
    
    train_df = df[df['date'].isin(train_dates)]
    test_df  = df[df['date'] == test_date]
    
    # skip if not enough data
    if train_df.empty or test_df.empty:
        continue

    # take average spot & rate for the training period
    S = train_df['spot'].iloc[-1]
    Ks = train_df['strike'].values
    Ts = train_df['T'].values
    r = train_df['r'].iloc[-1]
    market_prices = train_df['mid_price'].values

    # --- calibrate sigma using all data up to i ---
    res = minimize(objective, np.log(0.2),
                   args=(S, Ks, Ts, r, market_prices),
                   method='Nelder-Mead')
    sigma_hat_opt = res.x[0]
    sigma_opt = np.exp(sigma_hat_opt)

    # --- predict day i+1 ---
    S_test = test_df['spot'].iloc[0]
    Ks_test = test_df['strike'].values
    Ts_test = test_df['T'].values
    r_test = test_df['r'].iloc[0]
    market_prices_test = test_df['mid_price'].values

    model_prices_test = np.array([bs_call_price(S_test, K, T, r_test, sigma_opt)
                                  for K, T in zip(Ks_test, Ts_test)])
    rmse = np.sqrt(np.mean((model_prices_test - market_prices_test)**2))

    results.append({
        'train_end': train_dates[-1],
        'test_date': test_date,
        'sigma': sigma_opt,
        'rmse': rmse
    })

# --- Collect results ---
results_df = pd.DataFrame(results)
print(results_df.head())

# --- Plot rolling sigma and RMSE ---
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax1.plot(results_df['test_date'], results_df['sigma'], label='σ̂ (fitted)', color='red')
ax1.set_ylabel('Volatility', color='red')

ax2 = ax1.twinx()
ax2.plot(results_df['test_date'], results_df['rmse'], label='Out-of-sample RMSE', color='blue')
ax2.set_ylabel('RMSE', color='blue')

plt.title('Rolling Black–Scholes Calibration (Expanding Window)')
plt.show()