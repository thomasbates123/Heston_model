import numpy as np
import pandas as pd
from scipy.stats import norm
import sympy as sp
from typing import Literal, Dict
from dataclasses import dataclass


@dataclass
class OptionParameters:
    """Data class to hold option parameters"""
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to maturity (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility
    option_type: Literal['call', 'put']


class BlackScholesSymbolic:
    """
    Symbolic Black-Scholes model using SymPy for Greek calculation.
    Derives Greeks analytically using symbolic differentiation.
    """
    
    def __init__(self):
        # Define symbolic variables
        self.S, self.K, self.T, self.r, self.sigma = sp.symbols(
            'S K T r sigma', real=True, positive=True
        )
        
        # Define d1 and d2
        self.d1 = (sp.ln(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (
            self.sigma * sp.sqrt(self.T)
        )
        self.d2 = self.d1 - self.sigma * sp.sqrt(self.T)
        
        # Define pricing formulas
        self.call_price = self.S * sp.erf(self.d1 / sp.sqrt(2)) / 2 + self.S / 2 - \
                         self.K * sp.exp(-self.r * self.T) * (sp.erf(self.d2 / sp.sqrt(2)) / 2 + sp.Rational(1, 2))
        
        self.put_price = self.K * sp.exp(-self.r * self.T) * (sp.erf(-self.d2 / sp.sqrt(2)) / 2 + sp.Rational(1, 2)) - \
                        self.S * (sp.erf(-self.d1 / sp.sqrt(2)) / 2 + sp.Rational(1, 2))
        
        # Pre-compute symbolic Greeks
        self._compute_symbolic_greeks()
        
    def _compute_symbolic_greeks(self):
        """Compute all Greeks symbolically"""
        # Call Greeks
        self.call_delta_sym = sp.diff(self.call_price, self.S)
        self.call_gamma_sym = sp.diff(self.call_delta_sym, self.S)
        self.call_vega_sym = sp.diff(self.call_price, self.sigma)
        self.call_theta_sym = sp.diff(self.call_price, self.T)
        self.call_rho_sym = sp.diff(self.call_price, self.r)
        
        # Put Greeks
        self.put_delta_sym = sp.diff(self.put_price, self.S)
        self.put_gamma_sym = sp.diff(self.put_delta_sym, self.S)
        self.put_vega_sym = sp.diff(self.put_price, self.sigma)
        self.put_theta_sym = sp.diff(self.put_price, self.T)
        self.put_rho_sym = sp.diff(self.put_price, self.r)
        
        # Lambdify for fast numerical evaluation
        vars_list = [self.S, self.K, self.T, self.r, self.sigma]
        
        self.call_delta_func = sp.lambdify(vars_list, self.call_delta_sym, 'numpy')
        self.call_gamma_func = sp.lambdify(vars_list, self.call_gamma_sym, 'numpy')
        self.call_vega_func = sp.lambdify(vars_list, self.call_vega_sym, 'numpy')
        self.call_theta_func = sp.lambdify(vars_list, self.call_theta_sym, 'numpy')
        self.call_rho_func = sp.lambdify(vars_list, self.call_rho_sym, 'numpy')
        
        self.put_delta_func = sp.lambdify(vars_list, self.put_delta_sym, 'numpy')
        self.put_gamma_func = sp.lambdify(vars_list, self.put_gamma_sym, 'numpy')
        self.put_vega_func = sp.lambdify(vars_list, self.put_vega_sym, 'numpy')
        self.put_theta_func = sp.lambdify(vars_list, self.put_theta_sym, 'numpy')
        self.put_rho_func = sp.lambdify(vars_list, self.put_rho_sym, 'numpy')


class GreeksCalculator:
    """
    Main class for calculating option Greeks using Black-Scholes model.
    Supports both symbolic (SymPy) and numerical approaches.
    """
    
    def __init__(self):
        self.bs_symbolic = BlackScholesSymbolic()
        
    def calculate_greeks(self, params: OptionParameters) -> Dict[str, float]:
        """
        Calculate all Greeks for given option parameters.
        
        Args:
            params: OptionParameters dataclass instance
            
        Returns:
            Dictionary containing all Greeks
        """
        args = (params.S, params.K, params.T, params.r, params.sigma)
        
        if params.option_type == 'call':
            greeks = {
                'delta': float(self.bs_symbolic.call_delta_func(*args)),
                'gamma': float(self.bs_symbolic.call_gamma_func(*args)),
                'vega': float(self.bs_symbolic.call_vega_func(*args)),
                'theta': float(self.bs_symbolic.call_theta_func(*args)),
                'rho': float(self.bs_symbolic.call_rho_func(*args))
            }
        else:  # put
            greeks = {
                'delta': float(self.bs_symbolic.put_delta_func(*args)),
                'gamma': float(self.bs_symbolic.put_gamma_func(*args)),
                'vega': float(self.bs_symbolic.put_vega_func(*args)),
                'theta': float(self.bs_symbolic.put_theta_func(*args)),
                'rho': float(self.bs_symbolic.put_rho_func(*args))
            }
            
        return greeks
    
    def get_symbolic_greek(self, greek: str, option_type: Literal['call', 'put']) -> sp.Expr:
        """
        Return symbolic expression for a specific Greek.
        
        Args:
            greek: Name of Greek ('delta', 'gamma', 'vega', 'theta', 'rho')
            option_type: 'call' or 'put'
            
        Returns:
            SymPy expression
        """
        greek_map = {
            'call': {
                'delta': self.bs_symbolic.call_delta_sym,
                'gamma': self.bs_symbolic.call_gamma_sym,
                'vega': self.bs_symbolic.call_vega_sym,
                'theta': self.bs_symbolic.call_theta_sym,
                'rho': self.bs_symbolic.call_rho_sym
            },
            'put': {
                'delta': self.bs_symbolic.put_delta_sym,
                'gamma': self.bs_symbolic.put_gamma_sym,
                'vega': self.bs_symbolic.put_vega_sym,
                'theta': self.bs_symbolic.put_theta_sym,
                'rho': self.bs_symbolic.put_rho_sym
            }
        }
        
        return greek_map[option_type][greek]


class PortfolioGreeks:
    """
    Calculate portfolio-level Greeks for multiple option positions.
    """
    
    def __init__(self):
        self.calculator = GreeksCalculator()
        
    def calculate_portfolio_greeks(
        self, 
        positions: list[tuple[OptionParameters, float]]
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for a portfolio.
        
        Args:
            positions: List of (OptionParameters, quantity) tuples
            
        Returns:
            Dictionary of portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        for params, quantity in positions:
            greeks = self.calculator.calculate_greeks(params)
            for key in portfolio_greeks:
                portfolio_greeks[key] += greeks[key] * quantity
                
        return portfolio_greeks


# Example usage
if __name__ == "__main__":
    # Initialize calculator
    calc = GreeksCalculator()
    
    # Define option parameters
    params = OptionParameters(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type='call'
    )
    
    # Calculate Greeks
    greeks = calc.calculate_greeks(params)
    
    print("Option Parameters:")
    print(f"  Spot Price: ${params.S}")
    print(f"  Strike: ${params.K}")
    print(f"  Time to Maturity: {params.T} years")
    print(f"  Risk-free Rate: {params.r*100}%")
    print(f"  Volatility: {params.sigma*100}%")
    print(f"  Option Type: {params.option_type}")
    print("\nGreeks:")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    
    # Show symbolic expression for Delta
    print("\n" + "="*50)
    print("Symbolic Delta Expression (Call):")
    print(sp.simplify(calc.get_symbolic_greek('delta', 'call')))
    
    # Portfolio example
    print("\n" + "="*50)
    print("Portfolio Greeks Example:")
    portfolio = PortfolioGreeks()
    
    positions = [
        (params, 10),  # Long 10 calls
        (OptionParameters(100, 95, 1.0, 0.05, 0.2, 'put'), -5)  # Short 5 puts
    ]
    
    portfolio_greeks = portfolio.calculate_portfolio_greeks(positions)
    print("\nPortfolio Greeks:")
    for greek, value in portfolio_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")