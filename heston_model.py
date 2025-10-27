"""
Heston Model Implementation for Option Pricing

This module implements the Heston stochastic volatility model for pricing European options.
The Heston model assumes volatility follows a mean-reverting square root process.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


class HestonModel:
    """
    Implementation of the Heston stochastic volatility model.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity (in years)
    v0 : float
        Initial variance
    kappa : float
        Mean reversion rate
    theta : float
        Long-term variance
    sigma : float
        Volatility of volatility
    rho : float
        Correlation between stock and volatility
    """
    
    def __init__(self, S0, K, r, T, v0, kappa, theta, sigma, rho):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
    
    def characteristic_function(self, phi, j):
        """
        Characteristic function for the Heston model.
        
        Parameters:
        -----------
        phi : complex
            Integration variable
        j : int
            1 or 2, determines which probability we're computing
        """
        if j == 1:
            u = 0.5
            b = self.kappa - self.rho * self.sigma
        else:
            u = -0.5
            b = self.kappa
        
        a = self.kappa * self.theta
        
        d = np.sqrt((self.rho * self.sigma * phi * 1j - b)**2 - 
                   self.sigma**2 * (2 * u * phi * 1j - phi**2))
        
        g = (b - self.rho * self.sigma * phi * 1j + d) / \
            (b - self.rho * self.sigma * phi * 1j - d)
        
        C = self.r * phi * 1j * self.T + \
            (a / self.sigma**2) * ((b - self.rho * self.sigma * phi * 1j + d) * self.T - 
                                   2 * np.log((1 - g * np.exp(d * self.T)) / (1 - g)))
        
        D = ((b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2) * \
            ((1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T)))
        
        return np.exp(C + D * self.v0 + 1j * phi * np.log(self.S0))
    
    def probability(self, j):
        """
        Compute probability P_j using numerical integration.
        
        Parameters:
        -----------
        j : int
            1 or 2, determines which probability
        """
        def integrand(phi):
            try:
                cf = self.characteristic_function(phi, j)
                numerator = np.exp(-1j * phi * np.log(self.K)) * cf
                denominator = 1j * phi
                return np.real(numerator / denominator)
            except:
                return 0
        
        integral, _ = quad(integrand, 0, 100)
        return 0.5 + (1 / np.pi) * integral
    
    def call_price(self):
        """
        Calculate European call option price using Heston model.
        
        Returns:
        --------
        float
            Call option price
        """
        P1 = self.probability(1)
        P2 = self.probability(2)
        
        call = self.S0 * P1 - self.K * np.exp(-self.r * self.T) * P2
        return call
    
    def put_price(self):
        """
        Calculate European put option price using put-call parity.
        
        Returns:
        --------
        float
            Put option price
        """
        call = self.call_price()
        put = call - self.S0 + self.K * np.exp(-self.r * self.T)
        return put
    
    @staticmethod
    def calibrate(market_prices, S0, K_list, r, T_list, strikes, option_type='call'):
        """
        Calibrate Heston model parameters to market prices.
        
        Parameters:
        -----------
        market_prices : array-like
            Observed market option prices
        S0 : float
            Current stock price
        K_list : array-like
            List of strike prices
        r : float
            Risk-free rate
        T_list : array-like
            List of maturities
        strikes : array-like
            Strike prices for each option
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        dict
            Calibrated parameters
        """
        def objective(params):
            v0, kappa, theta, sigma, rho = params
            
            # Parameter constraints
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0:
                return 1e10
            if abs(rho) >= 1:
                return 1e10
            if 2 * kappa * theta < sigma**2:  # Feller condition
                return 1e10
            
            error = 0
            for i, (K, T, market_price) in enumerate(zip(K_list, T_list, market_prices)):
                try:
                    model = HestonModel(S0, K, r, T, v0, kappa, theta, sigma, rho)
                    if option_type == 'call':
                        model_price = model.call_price()
                    else:
                        model_price = model.put_price()
                    error += (model_price - market_price)**2
                except:
                    error += 1e10
            
            return error
        
        # Initial guess
        x0 = [0.04, 2.0, 0.04, 0.3, -0.5]
        
        # Bounds for parameters
        bounds = [(1e-6, 1), (1e-6, 10), (1e-6, 1), (1e-6, 2), (-0.99, 0.99)]
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        return {
            'v0': result.x[0],
            'kappa': result.x[1],
            'theta': result.x[2],
            'sigma': result.x[3],
            'rho': result.x[4]
        }
