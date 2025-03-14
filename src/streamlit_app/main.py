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
import pullback
import crypto_data
import equities_data

# Page config
st.set_page_config(
    page_title="B.A.A.S",
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

# EMA period
ema_period = st.sidebar.slider(
    "EMA Period",
    min_value=5,
    max_value=50,
    value=21,
    help="Period for Exponential Moving Average calculation"
)

# Pullback parameters
min_pullback_candles = st.sidebar.slider(
    "Minimum Pullback Candles",
    min_value=2,
    max_value=14,
    value=5,
    help="Minimum number of candles for pullback pattern"
)

# Breakout parameters
breakout_threshold = st.sidebar.slider(
    "Breakout Threshold (%)",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Minimum percentage move required for breakout confirmation"
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

def plot_chart(data, symbol, signals_df):
    """Create an interactive chart with Plotly"""
    # Calculate EMAs for the chart
    data['ema'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    data['ema_2'] = data['Close'].ewm(span=ema_period*2, adjust=False).mean()

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

    # Add EMAs
    fig.add_trace(
        go.Scatter(x=data.index, y=data['ema'], name=f'EMA{ema_period}', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['ema_2'], name=f'EMA{ema_period*2}', line=dict(color='orange')),
        row=1, col=1
    )

    # Add signals
    if not signals_df.empty:
        long_signals = signals_df[signals_df['type'] == 'long']
        short_signals = signals_df[signals_df['type'] == 'short']
        
        fig.add_trace(
            go.Scatter(
                x=long_signals['index'],
                y=long_signals['price'],
                name='Long Signal',
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=short_signals['index'],
                y=short_signals['price'],
                name='Short Signal',
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
        title=f'{symbol} - EMA Pullback Analysis',
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
                    _, signals_df = pullback.detect_precise_ema_pullback(
                        symbol_data, 
                        ema_period=ema_period,
                        min_trend_bars=min_pullback_candles,
                        breakout_threshold=breakout_threshold/100  # Convert percentage to decimal
                    )
                    
                    if not signals_df.empty:
                        # Get all signals for this symbol
                        for _, signal in signals_df.iterrows():
                            signal_summary.append({
                                'Symbol': symbol,
                                'Signal': 'Buy' if signal['type'] == 'long' else 'Sell',
                                'Price': signal['price'],
                                'Volume': symbol_data.loc[signal['index'], 'Volume'],
                                'Date': signal['index']
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
        # Create a list of symbols with any signals
        active_symbols = list(set(item['Symbol'] for item in st.session_state.signal_summary))
        
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=active_symbols,
            help="Select a symbol to view its chart and signals"
        )
        
        # Get data for selected symbol
        symbol_data = st.session_state.market_data[selected_symbol].copy()
        
        # Get signals for this symbol from the signal summary
        symbol_signals = pd.DataFrame([
            signal for signal in st.session_state.signal_summary 
            if signal['Symbol'] == selected_symbol
        ])
        
        # Convert to the format expected by plot_chart
        if not symbol_signals.empty:
            signals_df = pd.DataFrame({
                'index': symbol_signals['Date'],
                'type': symbol_signals['Signal'].map({'Buy': 'long', 'Sell': 'short'}),
                'price': symbol_signals['Price']
            })
        else:
            signals_df = pd.DataFrame()
        
        # Plot chart
        fig = plot_chart(symbol_data, selected_symbol, signals_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        signals_df = pd.DataFrame(st.session_state.signal_summary)
        # Sort by date in descending order
        signals_df = signals_df.sort_values('Date', ascending=False)
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
