"""
CSV Processor - Main class combining all processing modules
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# Try relative imports first (when used as module), fall back to absolute imports
try:
    from .csv_loader import CSVLoader
    from .price_canonicalizer import PriceCanonicalizer
    from .epsilon_calculator import EpsilonCalculator
    from .data_validator import DataValidator
except ImportError:
    from csv_loader import CSVLoader
    from price_canonicalizer import PriceCanonicalizer
    from epsilon_calculator import EpsilonCalculator
    from data_validator import DataValidator


class CSVProcessor:
    """
    Main CSV processor for options data combining all modules.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize with path to CSV file.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        self.loader = CSVLoader(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self.flags: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data."""
        self.df = self.loader.load_data()
        return self.df
    
    def show_sample(self, n: int = 5):
        """Display sample rows."""
        self.loader.show_sample(n)
    
    def show_all_data(self):
        """Display entire DataFrame."""
        self.loader.show_all_data()
    
    def canonicalize_price_fields(self, tau: float = 0.25):
        """
        Canonicalize mid price based on bid-ask spread.
        
        Args:
            tau: Threshold multiplier
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        self.df, price_flags = PriceCanonicalizer.canonicalize_mid_price(self.df, tau)
        
        # Initialize or merge flags
        if self.flags is None:
            self.flags = price_flags
        else:
            self.flags = pd.concat([self.flags, price_flags], axis=1)
    
    def compute_adaptive_epsilon(self, **kwargs):
        """
        Compute adaptive epsilon.
        
        Args:
            **kwargs: Arguments passed to EpsilonCalculator
            
        Returns:
            Series or DataFrame with epsilon values
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        return EpsilonCalculator.compute_adaptive_epsilon(self.df, **kwargs)
    
    def basic_data_sanity_check(self, **kwargs):
        """
        Perform comprehensive data quality checks.
        
        Args:
            **kwargs: Arguments passed to DataValidator
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        self.df, validation_flags = DataValidator.validate_data(self.df, **kwargs)
        
        # Merge validation flags
        if self.flags is None:
            self.flags = validation_flags
        else:
            # Merge without duplicates
            for col in validation_flags.columns:
                if col not in self.flags.columns:
                    self.flags[col] = validation_flags[col]
    
    def show_flagged_rows(self, flag_name: str = 'MID_MISMATCH'):
        """
        Display rows that have been flagged.
        
        Args:
            flag_name: Name of the flag to check
        """
        if self.df is None or self.flags is None:
            print("No data or flags available.")
            return
        
        PriceCanonicalizer.show_flagged_rows(self.df, self.flags, flag_name)
    
    def get_eligible_data(self) -> pd.DataFrame:
        """
        Get only eligible rows.
        
        Returns:
            DataFrame with only eligible rows
        """
        if self.df is None:
            raise RuntimeError("Data not loaded.")
        
        if 'eligible' not in self.df.columns:
            print("Warning: 'eligible' column not found. Returning all data.")
            return self.df
        
        return self.df[self.df['eligible']].copy()
    
    def get_ineligible_data(self) -> pd.DataFrame:
        """
        Get only ineligible rows.
        
        Returns:
            DataFrame with only ineligible rows
        """
        if self.df is None:
            raise RuntimeError("Data not loaded.")
        
        if 'eligible' not in self.df.columns:
            print("Warning: 'eligible' column not found. Returning empty DataFrame.")
            return pd.DataFrame()
        
        return self.df[~self.df['eligible']].copy()
