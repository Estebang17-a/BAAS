# Chart Pattern Screener

A powerful chart pattern screening application built with Streamlit that helps you identify trading opportunities in both crypto and equity markets.

## Features

- Screen multiple markets simultaneously (Crypto and Equities)
- Interactive charts with Plotly
- Support and Resistance trendline analysis
- EMA (Exponential Moving Average) indicator
- Buy/Sell signals based on pattern recognition
- Multiple timeframe analysis (1d, 4h, 1h)
- Customizable parameters
- Real-time data from multiple sources

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run src/streamlit_app/main.py
```

2. Use the sidebar controls to:
   - Select markets (Crypto/Equities)
   - Choose timeframe
   - Adjust lookback period
   - Modify EMA period

3. Click "Run Screening" to start the analysis

4. View results in two tabs:
   - Charts: Interactive chart view for individual symbols
   - Signals Summary: Table of current buy/sell signals

## Trading Strategy

The screener identifies potential trading opportunities based on the following conditions:

### Long (Buy) Signals
- Price is above EMA
- Resistance trendline has a negative slope
- Price breaks above resistance

### Short (Sell) Signals
- Price is below EMA
- Support trendline has a positive slope
- Price breaks below support

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- yfinance
- ccxt

## Disclaimer

This tool is for educational and research purposes only. Always conduct your own analysis and due diligence before making any investment decisions.
