import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Initialize session state variables
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'signal_summary' not in st.session_state:
    st.session_state.signal_summary = None
if 'has_run' not in st.session_state:
    st.session_state.has_run = False

# Add the parent directory to the path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trendline
import crypto_data
import equities_data

# Page config
st.set_page_config(
    page_title="Chart Pattern Screener",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📊 Balín Asset Analysis System")
st.markdown("""
This app helps you screen for chart patterns across both crypto and equity markets.
It uses trendline analysis and technical indicators to identify potential trading opportunities.
""")

# Sidebar controls
st.sidebar.header("Screening Parameters")

# Market selection
market_type = st.sidebar.multiselect(
    "Select Markets",
    ["Crypto", "Equities"],
    default=["Crypto"]
)

# Timeframe and period selection
st.sidebar.markdown("### Data Retrieval Parameters")

# Equities parameters
if "Equities" in market_type:
    st.sidebar.markdown("#### Equities Data")
    equities_timeframe = st.sidebar.selectbox(
        "Equities Timeframe",
        ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"],
        index=7,  # Default to 1d
        help="Timeframe for equities data retrieval (yfinance compatible)"
    )
    
    # Add timeframe description
    timeframe_descriptions = {
        "1m": "1 minute (max 7 days)",
        "2m": "2 minutes (max 60 days)",
        "5m": "5 minutes (max 60 days)",
        "15m": "15 minutes (max 60 days)",
        "30m": "30 minutes (max 60 days)",
        "60m": "1 hour (max 730 days)",
        "90m": "90 minutes (max 60 days)",
        "1d": "1 day (unlimited)",
        "5d": "5 days (unlimited)",
        "1wk": "1 week (unlimited)",
        "1mo": "1 month (unlimited)",
        "3mo": "3 months (unlimited)",
    }
    
    st.sidebar.markdown(f"*{timeframe_descriptions[equities_timeframe]}*")
    
    # Period selection based on timeframe type
    if equities_timeframe in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        # For intraday data, we use days as period
        period_options = ["1d", "5d", "1mo", "3mo"]
        period_descriptions = {
            "1d": "1 day",
            "5d": "5 days",
            "1mo": "1 month (about 30 days)",
            "3mo": "3 months (about 90 days)"
        }
    else:
        # For daily and above data, we can use longer periods
        period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"]
        period_descriptions = {
            "1d": "1 day",
            "5d": "5 days",
            "1mo": "1 month",
            "3mo": "3 months",
            "6mo": "6 months",
            "1y": "1 year",
            "2y": "2 years"
        }
    
    equities_periods = st.sidebar.selectbox(
        "Equities Period",
        options=period_options,
        index=2,  # Default to "1mo"
        format_func=lambda x: f"{x} ({period_descriptions[x]})",
        help="Historical data period to retrieve"
    )

# Crypto parameters
if "Crypto" in market_type:
    st.sidebar.markdown("#### Crypto Data")
    crypto_timeframe = st.sidebar.selectbox(
        "Crypto Timeframe",
        ["1d", "4h", "1h", "15m"],
        index=0,
        help="Timeframe for crypto data retrieval"
    )
    
    crypto_periods = st.sidebar.slider(
        "Crypto Candles",
        min_value=15,
        max_value=1000,
        value=100,
        help="Number of candles to retrieve for crypto data"
    )

# Analysis parameters
st.sidebar.markdown("### Analysis Parameters")

# Lookback period
lookback = st.sidebar.slider(
    "Lookback Period",
    min_value=10,
    max_value=50,
    value=14,
    help="Number of candles to analyze for pattern detection"
)

# EMA period
ema_period = st.sidebar.slider(
    "EMA Period",
    min_value=5,
    max_value=50,
    value=21,
    help="Period for Exponential Moving Average calculation"
)

def load_market_data():
    """Load market data based on user selection"""
    combined_data = None
    
    if "Crypto" in market_type:
        try:
            crypto_df = crypto_data.download_crypto_ohlc_data(
                timeframe=crypto_timeframe, 
                periods=crypto_periods
            )
            if combined_data is None:
                combined_data = crypto_df
            else:
                combined_data = combine_market_data(crypto_df=crypto_df, equities_df=combined_data)
        except Exception as e:
            st.sidebar.warning(f"Error loading crypto data: {str(e)}")
            
    if "Equities" in market_type:
        try:
            equities_df = equities_data.download_equities_ohlc_data(
                timeframe=equities_timeframe, 
                period=equities_periods
            )
            if combined_data is None:
                combined_data = equities_df
            else:
                combined_data = combine_market_data(crypto_df=combined_data, equities_df=equities_df)
        except Exception as e:
            st.sidebar.warning(f"Error loading equities data: {str(e)}")
    
    return combined_data

def combine_market_data(crypto_df=None, equities_df=None):
    """Combine market data from different sources"""
    dfs_to_concat = []
    
    if crypto_df is not None:
        if not isinstance(crypto_df.columns, pd.MultiIndex):
            raise ValueError("Crypto DataFrame must have MultiIndex columns")
        dfs_to_concat.append(crypto_df)
    
    if equities_df is not None:
        if not isinstance(equities_df.columns, pd.MultiIndex):
            raise ValueError("Equities DataFrame must have MultiIndex columns")
        dfs_to_concat.append(equities_df)
    
    if not dfs_to_concat:
        raise ValueError("At least one DataFrame must be provided")
    
    combined_df = pd.concat(dfs_to_concat, axis=1)
    combined_df = combined_df.sort_index(axis=1)
    
    return combined_df

def trendline_breakout_hl(data: pd.DataFrame, lookback: int, ema_period: int = 21):
    """Calculate trendline breakouts with specific pattern conditions"""
    # Calculate EMA
    data['EMA21'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Initialize arrays
    n = len(data)
    s_tl = np.full(n, np.nan)
    r_tl = np.full(n, np.nan)
    signals = np.zeros(n)
    r_slopes = np.full(n, np.nan)
    s_slopes = np.full(n, np.nan)
    
    for i in range(lookback, n):
        try:
            # Get window data
            window_close = data.iloc[i - lookback:i]['Close'].to_numpy()
            window_high = data.iloc[i - lookback:i]['High'].to_numpy()
            window_low = data.iloc[i - lookback:i]['Low'].to_numpy()
            
            # Calculate trendlines
            s_coefs, r_coefs = trendline.fit_trendlines_high_low(
                window_high, window_low, window_close
            )
            
            # Store slopes
            s_slopes[i] = s_coefs[0]
            r_slopes[i] = r_coefs[0]
            
            # Project trendlines to current bar
            s_val = s_coefs[1] + lookback * s_coefs[0]
            r_val = r_coefs[1] + lookback * r_coefs[0]
            
            s_tl[i] = s_val
            r_tl[i] = r_val
            
            # Get current price and EMA
            current_close = data.iloc[i]['Close']
            current_ema = data.iloc[i]['EMA21']

            # Average price condition
            window_average = window_close.mean()
            
            # Long condition: price > EMA + negative resistance slope
            if (window_average > current_ema) and (r_coefs[0] < 0) and (current_close > r_val):
                signals[i] = 1.0
                
            # Short condition: price < EMA + positive resistance slope    
            elif (window_average < current_ema) and (r_coefs[0] > 0) and (current_close < s_val):
                signals[i] = -1.0
                
            else:
                signals[i] = 0
                
        except Exception as e:
            signals[i] = signals[i - 1]
            continue
    
    return s_tl, r_tl, signals, data['EMA21'].to_numpy(), r_slopes, s_slopes

def plot_chart(data, symbol, s_tl, r_tl, signals, ema):
    """Create an interactive chart with Plotly"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add trendlines
    fig.add_trace(
        go.Scatter(x=data.index, y=s_tl, name='Support', line=dict(color='green', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=r_tl, name='Resistance', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Add EMA
    fig.add_trace(
        go.Scatter(x=data.index, y=ema, name=f'EMA{ema_period}', line=dict(color='blue')),
        row=1, col=1
    )

    # Add signals
    buy_signals = data[signals == 1].index
    sell_signals = data[signals == -1].index
    
    fig.add_trace(
        go.Scatter(
            x=buy_signals,
            y=data.loc[buy_signals, 'Low'],
            name='Buy Signal',
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sell_signals,
            y=data.loc[sell_signals, 'High'],
            name='Sell Signal',
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
        ),
        row=1, col=1
    )

    # Volume bars
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume'),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{symbol} - Chart Pattern Analysis',
        yaxis_title='Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig

# Main app logic
if st.sidebar.button("Run Screening") or not st.session_state.has_run:
    with st.spinner("Loading market data..."):
        try:
            st.session_state.market_data = load_market_data()
            
            if st.session_state.market_data is not None:
                st.success("Data loaded successfully!")
                st.session_state.has_run = True
                
                # Process all symbols first to get signals
                signal_summary = []
                symbols = list(st.session_state.market_data.columns.get_level_values(0).unique())
                
                for symbol in symbols:
                    symbol_data = st.session_state.market_data[symbol].copy()
                    _, _, signals, _, _, _ = trendline_breakout_hl(
                        symbol_data, lookback, ema_period
                    )
                    
                    last_signal = signals[-1]
                    if last_signal != 0:
                        signal_summary.append({
                            'Symbol': symbol,
                            'Signal': 'Buy' if last_signal == 1 else 'Sell',
                            'Price': symbol_data['Close'].iloc[-1],
                            'Volume': symbol_data['Volume'].iloc[-1]
                        })
                
                st.session_state.signal_summary = signal_summary

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.has_run = False

# Create tabs for different views
tab1, tab2 = st.tabs(["Charts", "Signals Summary"])

# Only show content if we have data
if st.session_state.market_data is not None and st.session_state.signal_summary:
    with tab1:
        # Create a list of symbols with active signals
        active_symbols = [item['Symbol'] for item in st.session_state.signal_summary]
        signal_types = {item['Symbol']: item['Signal'] for item in st.session_state.signal_summary}
        
        # Add signal type to symbol display
        symbol_options = [f"{symbol} ({signal_types[symbol]})" for symbol in active_symbols]
        
        # Symbol selector with signal information
        selected_display = st.selectbox(
            "Select Symbol (Showing only symbols with active signals)",
            options=symbol_options,
            format_func=lambda x: x
        )
        
        # Extract symbol from display string
        selected_symbol = selected_display.split(' (')[0]
        
        # Get data for selected symbol
        symbol_data = st.session_state.market_data[selected_symbol].copy()
        
        # Calculate signals
        s_tl, r_tl, signals, ema, r_slopes, s_slopes = trendline_breakout_hl(
            symbol_data, lookback, ema_period
        )
        
        # Plot chart
        fig = plot_chart(symbol_data, selected_symbol, s_tl, r_tl, signals, ema)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        signals_df = pd.DataFrame(st.session_state.signal_summary)
        st.dataframe(
            signals_df.style.apply(
                lambda x: ['background-color: lightgreen' if v == 'Buy' else 'background-color: lightcoral' for v in x],
                subset=['Signal']
            ),
            use_container_width=True
        )
elif st.session_state.has_run:
    st.info("No signals found for the current parameters.")
else:
    st.info("Click 'Run Screening' to start the analysis.") 