# Heston Model Options Arbitrage Investigation

An investigation into options arbitrage detection using the Heston stochastic volatility model with data from WRDS (Wharton Research Data Services).

## Overview

This project implements:
- **Heston Model**: A stochastic volatility model for pricing European options
- **WRDS Integration**: Connects to WRDS OptionMetrics for real market data
- **Arbitrage Detection**: Identifies multiple types of arbitrage opportunities including:
  - Put-Call Parity violations
  - Butterfly spread arbitrage
  - Calendar spread arbitrage
  - Box spread arbitrage

## Features

- 🎯 **Complete Heston Model Implementation**: Analytical pricing using characteristic functions
- 📊 **Model Calibration**: Calibrate model parameters to market prices
- 🔍 **Arbitrage Detection**: Multiple arbitrage detection algorithms
- 💾 **WRDS Integration**: Seamless connection to OptionMetrics database
- 📈 **Interactive Analysis**: Jupyter notebook for exploration and visualization
- 🧪 **Sample Data Generator**: Test without WRDS credentials

## Installation

```bash
# Clone the repository
git clone https://github.com/thomasbates123/Heston_model.git
cd Heston_model

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib (for visualization)
- WRDS Python library (optional, for real data)
- Jupyter (optional, for notebook)

## Quick Start

### Using Sample Data

```bash
python main.py --ticker AAPL
```

### Using WRDS Data

First, configure your WRDS credentials (see [WRDS Setup](#wrds-setup)).

```bash
python main.py --ticker AAPL --use-wrds --start-date 2023-01-01 --end-date 2023-12-31
```

### Using Jupyter Notebook

```bash
jupyter notebook arbitrage_analysis.ipynb
```

## Project Structure

```
Heston_model/
├── heston_model.py          # Heston model implementation
├── wrds_connector.py        # WRDS data connector
├── arbitrage_detector.py    # Arbitrage detection algorithms
├── main.py                  # Main analysis script
├── arbitrage_analysis.ipynb # Interactive Jupyter notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Components

### 1. Heston Model (`heston_model.py`)

Implements the Heston (1993) stochastic volatility model:

```
dS_t = μS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
```

Features:
- Analytical European option pricing via characteristic functions
- Parameter calibration to market prices
- Put and call pricing

**Example:**
```python
from heston_model import HestonModel

model = HestonModel(
    S0=100,      # Stock price
    K=100,       # Strike
    r=0.05,      # Risk-free rate
    T=1.0,       # Time to maturity
    v0=0.04,     # Initial variance
    kappa=2.0,   # Mean reversion rate
    theta=0.04,  # Long-term variance
    sigma=0.3,   # Volatility of volatility
    rho=-0.5     # Correlation
)

call_price = model.call_price()
put_price = model.put_price()
```

### 2. WRDS Connector (`wrds_connector.py`)

Connects to WRDS OptionMetrics database to fetch options data.

**Example:**
```python
from wrds_connector import WRDSConnector

connector = WRDSConnector()
options_df = connector.fetch_option_data('AAPL', '2023-01-01', '2023-12-31')
stock_df = connector.fetch_underlying_price('AAPL', '2023-01-01', '2023-12-31')
connector.close()
```

For testing without WRDS:
```python
from wrds_connector import generate_sample_data

options_df, stock_df = generate_sample_data(ticker='AAPL', days=30)
```

### 3. Arbitrage Detector (`arbitrage_detector.py`)

Implements multiple arbitrage detection algorithms:

**Put-Call Parity**: Detects violations of C - P = S - Ke^(-rT)

**Butterfly Spreads**: Ensures C(K1) - 2C(K2) + C(K3) ≥ 0

**Calendar Spreads**: Verifies longer maturity options are more expensive

**Box Spreads**: Checks box spread value equals (K2-K1)e^(-rT)

**Example:**
```python
from arbitrage_detector import ArbitrageDetector

detector = ArbitrageDetector(
    options_df,
    stock_price=150,
    risk_free_rate=0.05,
    tolerance=0.01
)

results = detector.detect_all_arbitrage()
```

## WRDS Setup

To use real WRDS data, you need:

1. A WRDS account (institutional access required)
2. Configure credentials:

```bash
# Create .pgpass file
echo "wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD" > ~/.pgpass
chmod 600 ~/.pgpass
```

Or use the interactive setup:
```python
import wrds
db = wrds.Connection(wrds_username='your_username')
```

See [WRDS documentation](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/) for details.

## Usage Examples

### Basic Arbitrage Analysis

```python
from main import analyze_arbitrage

# Run complete analysis
results = analyze_arbitrage(
    ticker='AAPL',
    use_sample_data=True
)

# Access specific arbitrage types
pcp_violations = results['put_call_parity']
butterfly_arb = results['butterfly']
calendar_arb = results['calendar']
box_arb = results['box_spread']
```

### Calibrate Heston Model

```python
from heston_model import HestonModel

# Calibrate to market prices
params = HestonModel.calibrate(
    market_prices=[10.5, 8.2, 6.1],
    S0=100,
    K_list=[95, 100, 105],
    r=0.05,
    T_list=[0.25, 0.25, 0.25],
    strikes=[95, 100, 105],
    option_type='call'
)

print(f"Calibrated parameters: {params}")
```

### Custom Arbitrage Detection

```python
from arbitrage_detector import ArbitrageDetector

# Create detector
detector = ArbitrageDetector(
    options_df,
    stock_price=150,
    risk_free_rate=0.05,
    tolerance=0.01  # Minimum profit threshold
)

# Detect specific arbitrage types
pcp = detector.detect_put_call_parity_violations()
butterfly = detector.detect_butterfly_arbitrage()
calendar = detector.detect_calendar_arbitrage()
box = detector.detect_box_spread_arbitrage()
```

## Output

The analysis produces:

1. **Console Output**: Summary of detected arbitrage opportunities
2. **CSV Files**: Detailed results for each arbitrage type
3. **Visualizations**: (in Jupyter notebook) Volatility smiles, profit distributions

Example output:
```
======================================================================
ARBITRAGE DETECTION SUMMARY
======================================================================
Put Call Parity: 5 opportunities
  Total Expected Profit: $12.50
  Average Profit: $2.50
Butterfly: 0 opportunities
Calendar: 2 opportunities
  Total Expected Profit: $3.20
  Average Profit: $1.60
Box Spread: 1 opportunities
  Total Expected Profit: $0.85
  Average Profit: $0.85

Total Arbitrage Opportunities: 8
======================================================================
```

## Methodology

### Heston Model

The Heston model captures:
- **Stochastic volatility**: Volatility follows a mean-reverting process
- **Leverage effect**: Negative correlation between returns and volatility
- **Volatility smile**: Can reproduce market-observed volatility patterns

### Arbitrage Detection

Each arbitrage type is checked using:
1. **Theoretical relationships**: No-arbitrage conditions
2. **Bid-ask spreads**: Account for transaction costs
3. **Tolerance threshold**: Minimum profit for practical execution

## Limitations

- Assumes European-style options
- Does not account for:
  - Dividends (can be extended)
  - Market impact
  - Margin requirements
  - Early exercise (American options)
- WRDS access requires institutional subscription

## References

- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options". *Review of Financial Studies*, 6(2), 327-343.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. Pearson.
- WRDS OptionMetrics documentation

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Contact

For questions or issues, please open a GitHub issue or contact the repository owner.
