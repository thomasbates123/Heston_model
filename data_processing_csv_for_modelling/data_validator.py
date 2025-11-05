"""
Data Validator - Perform data quality checks and validation
"""

import pandas as pd
from typing import Tuple, cast

# Try relative import first (when used as module), fall back to absolute import
try:
    from .epsilon_calculator import EpsilonCalculator
except ImportError:
    from epsilon_calculator import EpsilonCalculator


class DataValidator:
    """
    Validate options data quality and flag issues.
    """
    
    @staticmethod
    def validate_data(df: pd.DataFrame,
                     wide_threshold: float = 0.5,
                     use_adaptive_epsilon: bool = True,
                     **epsilon_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to validate
            wide_threshold: Threshold for relative spread to flag as WIDE
            use_adaptive_epsilon: If True, compute adaptive epsilon
            **epsilon_kwargs: Additional arguments for epsilon calculation
            
        Returns:
            Tuple of (updated_df, flags_df)
        """
        # Ensure mid_used exists
        if 'mid_used' not in df.columns:
            print("mid_used not found, using mid column")
            df = df.copy()
            df['mid_used'] = df['mid']
        
        # Initialize flags
        flags = DataValidator._initialize_flags(df)
        
        # Compute epsilon
        epsilon, epsilon_upper = DataValidator._compute_epsilon(
            df, use_adaptive_epsilon, epsilon_kwargs
        )
        
        df_updated = df.copy()
        df_updated['epsilon'] = epsilon
        df_updated['epsilon_upper'] = epsilon_upper
        
        # Perform checks
        DataValidator._check_locked_crossed(df_updated, flags)
        DataValidator._check_non_negativity(df_updated, flags)
        DataValidator._check_intrinsic_bounds(df_updated, flags, epsilon)
        DataValidator._check_upper_bounds(df_updated, flags, epsilon_upper)
        DataValidator._check_wide_spreads(df_updated, flags, wide_threshold)
        
        # Set eligibility
        df_updated = DataValidator._set_eligibility(df_updated, flags)
        
        # Print summary
        DataValidator._print_summary(df_updated, flags, use_adaptive_epsilon, epsilon_kwargs)
        
        return df_updated, flags
    
    @staticmethod
    def _initialize_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Initialize flags DataFrame."""
        flags = pd.DataFrame(index=df.index)
        flags['LOCKED'] = False
        flags['CROSSED'] = False
        flags['NEG'] = False
        flags['INTRINSIC_VIOL'] = False
        flags['UPPERBOUND_VIOL'] = False
        flags['WIDE'] = False
        return flags
    
    @staticmethod
    def _compute_epsilon(df: pd.DataFrame, 
                        use_adaptive_epsilon: bool,
                        epsilon_kwargs: dict) -> Tuple[pd.Series, pd.Series]:
        """Compute epsilon values."""
        if use_adaptive_epsilon:
            print("\n--- Computing Adaptive Epsilon ---")
            # Ensure return_components is False to get Series
            epsilon_kwargs_copy = epsilon_kwargs.copy()
            epsilon_kwargs_copy['return_components'] = False
            epsilon_result = EpsilonCalculator.compute_adaptive_epsilon(df, **epsilon_kwargs_copy)
            # Cast to Series since we know return_components=False
            epsilon: pd.Series = cast(pd.Series, epsilon_result)
            
            # Compute capped epsilon for upper bound
            mid_price = (df['best_bid'] + df['best_offer']) / 2
            tick_opt = pd.Series(0.05, index=df.index)
            tick_opt[mid_price >= 3.0] = 0.10
            tick_opt[mid_price < 0.50] = 0.01
            
            spread = (df['best_offer'] - df['best_bid']).clip(lower=0)
            epsilon_upper: pd.Series = 2 * tick_opt + 0.05 * spread
            epsilon_upper = epsilon_upper.clip(upper=0.10)
        else:
            fixed_epsilon = epsilon_kwargs.get('epsilon', 1e-6)
            epsilon = pd.Series(fixed_epsilon, index=df.index)
            epsilon_upper = epsilon
            print(f"Using fixed epsilon: {epsilon.iloc[0]}")
        
        return epsilon, epsilon_upper
    
    @staticmethod
    def _check_locked_crossed(df: pd.DataFrame, flags: pd.DataFrame):
        """Check for locked/crossed markets."""
        spread = df['best_offer'] - df['best_bid']
        locked = spread <= 0
        crossed = spread < 0
        flags.loc[locked, 'LOCKED'] = True
        flags.loc[crossed, 'CROSSED'] = True
        print(f"\n--- Data Quality Checks ---")
        print(f"LOCKED (bid >= ask): {locked.sum()} rows")
        print(f"CROSSED (bid > ask): {crossed.sum()} rows")
    
    @staticmethod
    def _check_non_negativity(df: pd.DataFrame, flags: pd.DataFrame):
        """Check for negative or missing prices."""
        bid = df['best_bid']
        ask = df['best_offer']
        neg = (bid < 0) | (ask <= 0) | bid.isna() | ask.isna()
        flags.loc[neg, 'NEG'] = True
        print(f"NEG (negative or zero/missing prices): {neg.sum()} rows")
    
    @staticmethod
    def _check_intrinsic_bounds(df: pd.DataFrame, flags: pd.DataFrame, epsilon: pd.Series):
        """Check intrinsic value lower bounds."""
        S = df['S']
        K = df['strike_price']
        ask = df['best_offer']
        cp_flag = df['cp_flag'].str.upper()
        
        is_call = cp_flag.isin(['C', 'CALL'])
        call_intrinsic = is_call & (ask < (S - K).clip(lower=0) - epsilon)
        
        is_put = cp_flag.isin(['P', 'PUT'])
        put_intrinsic = is_put & (ask < (K - S).clip(lower=0) - epsilon)
        
        intrinsic_viol = call_intrinsic | put_intrinsic
        flags.loc[intrinsic_viol, 'INTRINSIC_VIOL'] = True
        print(f"INTRINSIC_VIOL: {intrinsic_viol.sum()} rows (Ask < Intrinsic - ε)")
        
        if intrinsic_viol.sum() > 0:
            print(f"  Calls: {call_intrinsic.sum()}, Puts: {put_intrinsic.sum()}")
    
    @staticmethod
    def _check_upper_bounds(df: pd.DataFrame, flags: pd.DataFrame, epsilon_upper: pd.Series):
        """Check upper bounds."""
        S = df['S']
        K = df['strike_price']
        bid = df['best_bid']
        cp_flag = df['cp_flag'].str.upper()
        
        is_call = cp_flag.isin(['C', 'CALL'])
        call_upper_viol = is_call & (bid > S + epsilon_upper)
        
        is_put = cp_flag.isin(['P', 'PUT'])
        put_upper_viol = is_put & (bid > K + epsilon_upper)
        
        upper_viol = call_upper_viol | put_upper_viol
        flags.loc[upper_viol, 'UPPERBOUND_VIOL'] = True
        print(f"UPPERBOUND_VIOL: {upper_viol.sum()} rows (Bid > Upper bound + ε_capped)")
        
        if upper_viol.sum() > 0:
            print(f"  Calls: {call_upper_viol.sum()}, Puts: {put_upper_viol.sum()}")
    
    @staticmethod
    def _check_wide_spreads(df: pd.DataFrame, flags: pd.DataFrame, wide_threshold: float):
        """Check for wide spreads."""
        spread = df['best_offer'] - df['best_bid']
        mid_calc = (df['best_bid'] + df['best_offer']) / 2
        rel_spread = spread / mid_calc.replace(0, 1e-12)
        
        wide = rel_spread > wide_threshold
        flags.loc[wide, 'WIDE'] = True
        print(f"WIDE (spread > {wide_threshold*100}%): {wide.sum()} rows")
    
    @staticmethod
    def _set_eligibility(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
        """Set eligibility based on flags."""
        df_updated = df.copy()
        ineligible = (flags['CROSSED'] | flags['LOCKED'] | flags['NEG'] | 
                     flags['INTRINSIC_VIOL'] | flags['UPPERBOUND_VIOL'])
        df_updated['eligible'] = ~ineligible
        return df_updated
    
    @staticmethod
    def _print_summary(df: pd.DataFrame, flags: pd.DataFrame, 
                      use_adaptive_epsilon: bool, epsilon_kwargs: dict):
        """Print eligibility summary."""
        n_eligible = df['eligible'].sum()
        n_total = len(df)
        print(f"\n--- Eligibility Summary ---")
        print(f"Eligible: {n_eligible} out of {n_total} ({n_eligible/n_total*100:.2f}%)")
        print(f"Ineligible: {n_total - n_eligible} ({(n_total - n_eligible)/n_total*100:.2f}%)")
        
        print(f"\nIneligibility breakdown:")
        for flag in ['CROSSED', 'LOCKED', 'NEG', 'INTRINSIC_VIOL', 'UPPERBOUND_VIOL']:
            print(f"  {flag}: {flags[flag].sum()}")
        
        # Show epsilon impact comparison if using adaptive epsilon
        if use_adaptive_epsilon:
            print(f"\n--- Epsilon Impact Analysis ---")
            
            # Get data
            S = df['S']
            K = df['strike_price']
            bid = df['best_bid']
            ask = df['best_offer']
            cp_flag = df['cp_flag'].str.upper()
            
            # Use small fixed epsilon for comparison
            fixed_epsilon_value = epsilon_kwargs.get('fixed_epsilon_comparison', 1e-6)
            epsilon_fixed = pd.Series(fixed_epsilon_value, index=df.index)
            
            # Recompute violations with fixed epsilon
            is_call = cp_flag.isin(['C', 'CALL'])
            is_put = cp_flag.isin(['P', 'PUT'])
            
            # Intrinsic violations with fixed epsilon
            call_intrinsic_fixed = is_call & (ask < (S - K).clip(lower=0) - epsilon_fixed)
            put_intrinsic_fixed = is_put & (ask < (K - S).clip(lower=0) - epsilon_fixed)
            intrinsic_viol_fixed = call_intrinsic_fixed | put_intrinsic_fixed
            
            # Upper bound violations with fixed epsilon
            call_upper_viol_fixed = is_call & (bid > S + epsilon_fixed)
            put_upper_viol_fixed = is_put & (bid > K + epsilon_fixed)
            upper_viol_fixed = call_upper_viol_fixed | put_upper_viol_fixed
            
            # Count prevented flags (false positives with fixed epsilon)
            prevented_intrinsic = (intrinsic_viol_fixed & ~flags['INTRINSIC_VIOL']).sum()
            prevented_upper = (upper_viol_fixed & ~flags['UPPERBOUND_VIOL']).sum()
            
            # Compute eligibility with fixed epsilon
            ineligible_fixed = (flags['CROSSED'] | flags['LOCKED'] | flags['NEG'] | 
                              intrinsic_viol_fixed | upper_viol_fixed)
            n_eligible_fixed = (~ineligible_fixed).sum()
            
            print(f"Comparison vs fixed ε={fixed_epsilon_value}:")
            print(f"\nLower bound violations (intrinsic):")
            print(f"  With fixed ε: {intrinsic_viol_fixed.sum()} violations")
            print(f"  With adaptive ε: {flags['INTRINSIC_VIOL'].sum()} violations")
            print(f"  False positives prevented: {prevented_intrinsic}")
            
            print(f"\nUpper bound violations:")
            print(f"  With fixed ε: {upper_viol_fixed.sum()} violations")
            print(f"  With adaptive ε: {flags['UPPERBOUND_VIOL'].sum()} violations")
            print(f"  False positives prevented: {prevented_upper}")
            
            print(f"\nEligibility comparison:")
            print(f"  With fixed ε={fixed_epsilon_value}: {n_eligible_fixed} ({n_eligible_fixed/n_total*100:.2f}%)")
            print(f"  With smart ε: {n_eligible} ({n_eligible/n_total*100:.2f}%)")
            print(f"  Additional eligible rows: {n_eligible - n_eligible_fixed}")
            
            # Show mean epsilon values used
            if 'epsilon' in df.columns:
                print(f"\nEpsilon statistics:")
                print(f"  Mean adaptive ε: ${df['epsilon'].mean():.6f}")
                print(f"  Median adaptive ε: ${df['epsilon'].median():.6f}")
                print(f"  Max adaptive ε: ${df['epsilon'].max():.6f}")
