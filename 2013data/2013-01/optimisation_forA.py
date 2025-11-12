import numpy as np
import pandas as pd
from pathlib import Path
# Fake market data
csv_path = Path(__file__).parent / '2013-01-02options.csv'
print('Loading:', csv_path)
df = pd.read_csv(csv_path)
print(df.head())

csv_path = Path(__file__).parent / '2013-01-02stocks.csv'
print('Loading:', csv_path)
df2 = pd.read_csv(csv_path)
print(df.head())

S0 = df2[df2['symbol']=='A']['close']
print(S0)

strikes = sorted(df[df['underlying']=='A']['strike'].unique())

df['expiration'] = pd.to_datetime(df['expiration'])
df['quote_date'] = pd.to_datetime(df['quote_date'])

# Time to maturity in years
df['T'] = (df['expiration'] - df['quote_date']).dt.days / 365

maturities = sorted(df[df['underlying']=='A']['T'].unique())

print(strikes)
print(maturities)

# Compute mid price
df['mid_price'] = (df['bid'] + df['ask']) / 2

# Filter call options for underlying A
df_call = df[(df['underlying']=='A') & (df['type']=='call')]

# Pivot table: rows = maturity, columns = strike
market_prices_matrix = df_call.pivot(index='T', columns='strike', values='mid_price')

# Convert to numpy array
market_prices = market_prices_matrix.values

print("Market Prices Matrix:")
print(market_prices)



# Suppose these are "observed market prices"
market_prices = np.array([
    [12, 8, 5],   # T=0.5
    [14, 10, 6]   # T=1.0
])



market_prices = []
print(df.columns)
exit()

#for i in in range(len(strikes)):


# Fake “model” function
def model_price(theta, K, T):
    a, b = theta  # pretend θ has 2 parameters only
    return a + b*(K-100) + 0.5*T


def objective(theta):
    err = 0.0
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            model_val = model_price(theta, K, T)
            err += (model_val - market_prices[i,j])**2
    return err


from scipy.optimize import minimize

theta0 = [1.0, 0.1]  # initial guess

res = minimize(objective, theta0, method='L-BFGS-B')

print("Estimated theta:", res.x)


for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        print(f"T={T}, K={K}, Market={market_prices[i,j]}, Model={model_price(res.x, K, T):.2f}")
