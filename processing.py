#2013#-01-02options
#2013-01-02stocks

import pandas as pd

options = pd.read_csv("2013-01/2013-01-02options.csv")
stocks = pd.read_csv("2013-01/2013-01-02stocks.csv")
spot = stocks[['symbol', 'close']]
spot = stocks[['symbol', 'close']].rename(columns={'symbol': 'underlying'})


options = options[options['type'] == 'call']
options['expiration'] = pd.to_datetime(options['expiration'])
options['quote_date'] = pd.to_datetime(options['quote_date'])

# Time to maturity in years
options['T'] = (options['expiration'] - options['quote_date']).dt.days / 365.0


options = pd.merge(options, spot, on='underlying', how='left')

options = options.groupby(['underlying', 'T'])


print('options\n',options)



import numpy as np

import QuantLib as ql

def heston_price(S, K, T, r, q, kappa, theta, sigma, rho, v0, option_type='call'):
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type=='call' else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(ql.Date().from_date(datetime.today() + timedelta(days=int(T*365))))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, ql.QuoteHandle(ql.SimpleQuote(r)), day_count))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, ql.QuoteHandle(ql.SimpleQuote(q)), day_count))
    process = ql.HestonProcess(r_handle, q_handle, spot_handle, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(ql.Date().from_date(datetime.today() + timedelta(days=int(T*365)))))
    option.setPricingEngine(engine)
    return option.NPV()


def objective_price(params, df):
    kappa, theta, sigma, rho, v0 = params
    total = 0.0
    for _, row in df.iterrows():
        C_model = heston_price(row['spot'], row['strike'], row['T'], 
                               row['r'], row['q'],
                               kappa, theta, sigma, rho, v0,
                               row['type'])
        total += row['w'] * (C_model - row['mid_price'])**2
    return total


bounds = [
    (0.01, 20.0),   # kappa
    (1e-6, 1.0),    # theta
    (1e-4, 3.0),    # sigma
    (-0.999, 0.999),# rho
    (1e-6, 1.0)     # v0
]

initial_guess = [1.0, 0.04, 0.5, -0.5, 0.04]


from scipy.optimize import differential_evolution, minimize

# global search
res_de = differential_evolution(lambda p: objective_price(p, options), bounds, maxiter=200, polish=False)

# local refinement
res_local = minimize(lambda p: objective_price(p, options), res_de.x, 
                     bounds=bounds, method='L-BFGS-B')

params_calibrated = res_local.x
print("Calibrated params [kappa, theta, sigma, rho, v0] =", params_calibrated)


options['model_price'] = options.apply(lambda row: heston_price(
    row['spot'], row['strike'], row['T'], row['r'], row['q'],
    *params_calibrated, row['type']), axis=1)


import matplotlib.pyplot as plt

plt.scatter(options['mid_price'], options['model_price'])
plt.xlabel("Market price")
plt.ylabel("Model price")
plt.title("Heston calibration fit")
plt.plot([0, options['mid_price'].max()], [0, options['mid_price'].max()], 'r--')
plt.show()


rmse = np.sqrt(np.mean((options['model_price'] - options['mid_price'])**2))
print("RMSE =", rmse)
