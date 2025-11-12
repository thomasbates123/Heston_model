import numpy as np

# Fake market data
S0 = 100
strikes = np.array([90, 100, 110])
maturities = np.array([0.5, 1.0])  # in years

# Suppose these are "observed market prices"
market_prices = np.array([
    [12, 8, 5],   # T=0.5
    [14, 10, 6]   # T=1.0
])


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
