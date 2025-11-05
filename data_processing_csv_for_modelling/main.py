"""
Main entry point for CSV processing pipeline
"""

import os
from pathlib import Path

# Try relative import first (when used as module), fall back to absolute import
try:
    from .csv_processor import CSVProcessor
except ImportError:
    from csv_processor import CSVProcessor


def main():
    """Run the complete CSV processing pipeline."""
    # SPECIFY CSV FILE PATH
    # Get the parent directory (Heston_model) and construct path to CSV
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    csv_file = parent_dir / "aapl_options_jan2023.csv"
    
    # Check if file exists
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        print(f"Current directory: {current_dir}")
        print(f"Looking for file in: {parent_dir}")
        print("\nPlease update the csv_file path in main.py to point to your CSV file.")
        return None
    
    # Initialize processor
    processor = CSVProcessor(str(csv_file))
    
    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    processor.load_data()
    processor.show_sample()
    
    # Step 2: Canonicalize prices
    print("\n" + "=" * 60)
    print("STEP 2: Canonicalizing Price Fields")
    print("=" * 60)
    processor.canonicalize_price_fields(tau=0.25)
    processor.show_flagged_rows('MID_MISMATCH')
    
    # Step 3: Perform data quality checks
    print("\n" + "=" * 60)
    print("STEP 3: Data Quality Validation")
    print("=" * 60)
    processor.basic_data_sanity_check(
        wide_threshold=0.5,
        use_adaptive_epsilon=True,
        dt_lat_seconds=1.0,
        fallback_iv=0.3
    )
    
    # Step 4: Show summary
    print("\n" + "=" * 60)
    print("STEP 4: Summary")
    print("=" * 60)
    
    # Show sample of ineligible data
    print("\n--- Sample Ineligible Rows ---")
    ineligible = processor.get_ineligible_data()
    if len(ineligible) > 0:
        cols = ['best_bid', 'best_offer', 'mid_used', 'S', 'strike_price', 'cp_flag', 'eligible']
        cols = [c for c in cols if c in ineligible.columns]
        print(ineligible[cols].head(10))
    else:
        print("No ineligible rows found.")
    
    # Get eligible data for further processing
    eligible_data = processor.get_eligible_data()
    print(f"\nFinal eligible dataset: {len(eligible_data)} rows")
    
    return processor


if __name__ == "__main__":
    processor = main()
