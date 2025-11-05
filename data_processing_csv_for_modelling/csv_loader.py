"""
CSV Loader - Load and inspect CSV data
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class CSVLoader:
    """
    Simple CSV loader for options data.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize with path to CSV file.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data into pandas DataFrame.
        
        Returns:
            Loaded DataFrame
        """
        print(f"Loading data from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def show_sample(self, n: int = 5):
        """
        Display sample rows from the data.
        
        Args:
            n: Number of rows to display
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print(f"\nFirst {n} rows:")
        print(self.df.head(n))

    def show_all_data(self):
        """
        Display the entire DataFrame.
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(f"\n--- Full DataFrame ({len(self.df)} rows) ---")
        print(self.df)

        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded DataFrame.
        
        Returns:
            DataFrame if loaded, None otherwise
        """
        return self.df
