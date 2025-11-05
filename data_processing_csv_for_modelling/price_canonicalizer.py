"""
Price Canonicalizer - Canonicalize mid prices based on bid-ask spread
"""

import pandas as pd
from typing import Tuple


class PriceCanonicalizer:
    """
    Canonicalize option prices and flag inconsistencies.
    """
    
    @staticmethod
    def canonicalize_mid_price(df: pd.DataFrame, 
                               tau: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Canonicalize mid price based on bid-ask spread.
        
        Args:
            df: DataFrame with price columns
            tau: Threshold multiplier (default 0.25)
            
        Returns:
            Tuple of (updated_df, flags_df)
        """
        # Initialize flags
        flags = pd.DataFrame(index=df.index)
        flags['MID_MISMATCH'] = False
        
        # Calculate theoretical midpoint
        mid_calc = (df['best_bid'] + df['best_offer']) / 2
        
        # Calculate spread
        spread = df['best_offer'] - df['best_bid']
        
        # Check if mid differs too much from mid_calc
        mid_diff = abs(df['mid'] - mid_calc)
        threshold = tau * spread
        
        # Flag mismatches
        mismatch = mid_diff > threshold
        flags.loc[mismatch, 'MID_MISMATCH'] = True
        
        # Create mid_used column
        df_updated = df.copy()
        df_updated['mid_used'] = df['mid'].copy()
        df_updated.loc[mismatch, 'mid_used'] = mid_calc[mismatch]
        
        # Print summary
        n_replaced = mismatch.sum()
        print(f"Mid price adjustments: {n_replaced} out of {len(df)} ({n_replaced/len(df)*100:.2f}%)")
        
        return df_updated, flags
    
    @staticmethod
    def show_flagged_rows(df: pd.DataFrame, 
                         flags: pd.DataFrame, 
                         flag_name: str = 'MID_MISMATCH'):
        """
        Display rows that have been flagged.
        
        Args:
            df: Main DataFrame
            flags: Flags DataFrame
            flag_name: Name of the flag to check
        """
        if flag_name not in flags.columns:
            print(f"Flag '{flag_name}' not found.")
            return

        flagged = df[flags[flag_name]]

        print(f"\n--- Flagged rows for '{flag_name}' ---")
        print(f"Total flagged: {len(flagged)} out of {len(df)} rows")

        if len(flagged) > 0:
            cols_to_show = ['best_bid', 'best_offer', 'mid', 'mid_used']
            cols_to_show = [col for col in cols_to_show if col in flagged.columns]

            print(f"\nFirst 10 flagged rows:")
            print(flagged[cols_to_show].head(10))
        else:
            print("No rows flagged.")
