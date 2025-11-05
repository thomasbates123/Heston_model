"""
Epsilon Calculator - Compute adaptive epsilon for option pricing tolerance
"""

import pandas as pd
from typing import Union, Optional


class EpsilonCalculator:
    """
    Calculate adaptive epsilon based on market microstructure.
    """
    
    @staticmethod
    def compute_adaptive_epsilon(df: pd.DataFrame,
                                dt_lat_seconds: float = 1.0,
                                fallback_iv: float = 0.3,
                                tick_multiplier: float = 2.0,
                                spread_multiplier: float = 0.10,
                                move_multiplier: float = 3.0,
                                manual_tick_size: Optional[float] = None,
                                near_expiry_boost: bool = True,
                                expiry_threshold_days: int = 7,
                                return_components: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute adaptive epsilon for each row.
        
        Args:
            df: DataFrame with options data
            dt_lat_seconds: Latency window in seconds
            fallback_iv: Fallback IV if all else fails
            tick_multiplier: Multiplier for tick component
            spread_multiplier: Multiplier for spread component
            move_multiplier: Multiplier for move component
            manual_tick_size: Override tick size inference (optional)
            near_expiry_boost: Apply boost for near-expiry options
            expiry_threshold_days: Days to expiry below which boost applies
            return_components: If True, return DataFrame with components
            
        Returns:
            Series or DataFrame with epsilon values (and components if requested)
        """
        # Convert latency to years
        dt_lat = dt_lat_seconds / (252 * 23400)
        
        # Get bid-ask spread
        spread = (df['best_offer'] - df['best_bid']).clip(lower=0)
        
        # Get underlying price
        S = df['S']
        
        # Get IV with robust fallback
        sigma = EpsilonCalculator._get_implied_volatility(df, fallback_iv)
        
        # Infer or use manual tick size
        tick_opt = EpsilonCalculator._infer_tick_size(df, manual_tick_size)
        
        # Compute three components
        discretization = tick_multiplier * tick_opt
        quote_noise = spread_multiplier * spread
        underlying_move = move_multiplier * S * sigma * (dt_lat ** 0.5)
        
        # Apply near-expiry boost if requested
        if near_expiry_boost and 'dte' in df.columns:
            near_expiry = df['dte'] < expiry_threshold_days
            boost_factor = 2.0
            underlying_move = underlying_move.where(~near_expiry, underlying_move * boost_factor)
            n_boosted = near_expiry.sum()
            if n_boosted > 0:
                print(f"Applied near-expiry boost to {n_boosted} rows (DTE < {expiry_threshold_days})")
        
        # Ensure all components are finite and non-negative
        discretization = discretization.replace([float('inf'), float('-inf')], 0).clip(lower=0)
        quote_noise = quote_noise.replace([float('inf'), float('-inf')], 0).clip(lower=0)
        underlying_move = underlying_move.replace([float('inf'), float('-inf')], 0).clip(lower=0)
        
        # Take maximum
        epsilon = pd.concat([discretization, quote_noise, underlying_move], axis=1).max(axis=1)
        
        # Print statistics
        EpsilonCalculator._print_epsilon_stats(epsilon, discretization, quote_noise, underlying_move)
        
        if return_components:
            dominant_component = pd.DataFrame({
                'discretization': discretization,
                'quote_noise': quote_noise,
                'underlying_move': underlying_move
            }).idxmax(axis=1)
            
            result = pd.DataFrame({
                'epsilon': epsilon,
                'eps_discretization': discretization,
                'eps_quote_noise': quote_noise,
                'eps_underlying_move': underlying_move,
                'eps_dominant': dominant_component
            }, index=df.index)
            return result
        else:
            return epsilon
    
    @staticmethod
    def _get_implied_volatility(df: pd.DataFrame, fallback_iv: float) -> pd.Series:
        """Get implied volatility with fallback hierarchy."""
        sigma = None
        for col_name in ['impl_volatility', 'implied_volatility', 'iv']:
            if col_name in df.columns:
                sigma = df[col_name].copy()
                break
        
        if sigma is None:
            sigma = pd.Series(fallback_iv, index=df.index)
            print(f"Warning: No IV column found. Using global fallback IV={fallback_iv}")
        else:
            if 'expiration' in df.columns:
                expiry_medians = sigma.groupby(df['expiration']).transform('median')
                sigma = sigma.fillna(expiry_medians)
            
            n_nans_before = sigma.isna().sum()
            sigma = sigma.fillna(fallback_iv)
            
            if n_nans_before > 0:
                print(f"Filled {n_nans_before} NaN IVs (expiry median + global fallback)")
        
        sigma = sigma.clip(lower=0.01).replace([float('inf'), float('-inf')], fallback_iv)
        return sigma
    
    @staticmethod
    def _infer_tick_size(df: pd.DataFrame, manual_tick_size: Optional[float] = None) -> pd.Series:
        """Infer tick size from option prices."""
        if manual_tick_size is not None:
            tick_opt = pd.Series(manual_tick_size, index=df.index)
            print(f"Using manual tick size: ${manual_tick_size}")
        else:
            mid_price = (df['best_bid'] + df['best_offer']) / 2
            
            tick_opt = pd.Series(0.05, index=df.index)
            tick_opt[mid_price >= 3.0] = 0.10
            tick_opt[mid_price < 0.50] = 0.01
            
            print(f"Inferred tick sizes: {tick_opt.value_counts().to_dict()}")
        
        return tick_opt
    
    @staticmethod
    def _print_epsilon_stats(epsilon: pd.Series, 
                            discretization: pd.Series,
                            quote_noise: pd.Series,
                            underlying_move: pd.Series):
        """Print epsilon statistics."""
        print(f"\n--- Adaptive Epsilon Statistics ---")
        print(f"Mean epsilon: ${epsilon.mean():.4f}")
        print(f"Median epsilon: ${epsilon.median():.4f}")
        print(f"Min epsilon: ${epsilon.min():.4f}")
        print(f"Max epsilon: ${epsilon.max():.4f}")
        print(f"Std epsilon: ${epsilon.std():.4f}")
        print(f"\nComponent breakdown (mean):")
        print(f"  Discretization: ${discretization.mean():.4f} ({(discretization == epsilon).sum()} rows dominate)")
        print(f"  Quote noise:    ${quote_noise.mean():.4f} ({(quote_noise == epsilon).sum()} rows dominate)")
        print(f"  Underlying move: ${underlying_move.mean():.4f} ({(underlying_move == epsilon).sum()} rows dominate)")
        
        dominant_component = pd.DataFrame({
            'discretization': discretization,
            'quote_noise': quote_noise,
            'underlying_move': underlying_move
        }).idxmax(axis=1)
        print(f"\nDominant component distribution:")
        print(dominant_component.value_counts())
