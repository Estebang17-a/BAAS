import pandas as pd
import numpy as np

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def detect_precise_ema_pullback(df, ema_period=21, min_trend_bars=5, breakout_threshold=0.001):
    """
    Detect precise EMA pullback pattern with breakout confirmation
    
    Parameters:
    df: DataFrame with OHLC data
    ema_period: Period for EMA (default 21)
    min_trend_bars: Minimum bars to confirm trend
    breakout_threshold: Minimum price movement to confirm breakout
    """
    df_copy = df.copy()
    
    # Calculate EMAs
    df_copy['ema'] = calculate_ema(df_copy['Close'], ema_period)
    
    # Calculate EMA slope using rolling window
    df_copy['ema_slope'] = df_copy['ema'].diff(min_trend_bars) / min_trend_bars
    
    # Initialize arrays for pattern detection
    signals = []
    
    for i in range(min_trend_bars + 10, len(df_copy)):
        window = df_copy.iloc[i-10:i+1]  # Look at recent price action
        current_bar = df_copy.iloc[i]
        prev_bar = df_copy.iloc[i-1]
        
        # Check for long setup
        if (current_bar['ema_slope'] > 0):  # Uptrend
            # Check for pullback trend (series of lower highs and lower lows)
            pullback_window = window.iloc[-5:-1]  # Look at last 4 bars before current
            has_pullback_trend = (
                (pullback_window['High'].diff().mean() < 0) and  # Average declining highs
                (pullback_window['Low'].diff().mean() < 0) and   # Average declining lows
                (pullback_window['Close'].iloc[-1] < pullback_window['Close'].iloc[0])  # Overall down move
            )
            
            # Check if we had a pullback to EMA
            touched_ema = any(
                (window['Low'].iloc[-4:-1] <= window['ema'].iloc[-4:-1]) &  # Price touched/crossed EMA
                (window['High'].iloc[-4:-1] >= window['ema'].iloc[-4:-1])    # in recent bars
            )
            
            # Confirm price was above EMA before pullback
            pre_pullback_trend = all(
                window['Close'].iloc[-8:-4] > window['ema'].iloc[-8:-4]
            )
            
            # Check for breakout candle
            breakout = (
                current_bar['Close'] > prev_bar['High'] and          # Breaks previous high
                current_bar['Close'] > current_bar['Open'] and       # Bullish candle
                current_bar['Close'] > current_bar['ema'] and        # Closes above EMA
                (current_bar['Close'] - prev_bar['High']) / prev_bar['High'] > breakout_threshold  # Significant breakout
            )
            
            if touched_ema and pre_pullback_trend and breakout and has_pullback_trend:
                signals.append({
                    'index': window.index[-1],
                    'type': 'long',
                    'price': current_bar['Close'],
                    'ema': current_bar['ema']
                })
        
        # Check for short setup
        elif (current_bar['ema_slope'] < 0):  # Downtrend
            # Check for pullback trend (series of higher highs and higher lows)
            pullback_window = window.iloc[-5:-1]  # Look at last 4 bars before current
            has_pullback_trend = (
                (pullback_window['High'].diff().mean() > 0) and  # Average rising highs
                (pullback_window['Low'].diff().mean() > 0) and   # Average rising lows
                (pullback_window['Close'].iloc[-1] > pullback_window['Close'].iloc[0])  # Overall up move
            )
            
            # Check if we had a pullback to EMA
            touched_ema = any(
                (window['High'].iloc[-4:-1] >= window['ema'].iloc[-4:-1]) &  # Price touched/crossed EMA
                (window['Low'].iloc[-4:-1] <= window['ema'].iloc[-4:-1])     # in recent bars
            )
            
            # Confirm price was below EMA before pullback
            pre_pullback_trend = all(
                window['Close'].iloc[-8:-4] < window['ema'].iloc[-8:-4]
            )
            
            # Check for breakdown candle
            breakdown = (
                current_bar['Close'] < prev_bar['Low'] and          # Breaks previous low
                current_bar['Close'] < current_bar['Open'] and      # Bearish candle
                current_bar['Close'] < current_bar['ema'] and       # Closes below EMA
                (prev_bar['Low'] - current_bar['Close']) / prev_bar['Low'] > breakout_threshold  # Significant breakdown
            )
            
            if touched_ema and pre_pullback_trend and breakdown and has_pullback_trend:
                signals.append({
                    'index': window.index[-1],
                    'type': 'short',
                    'price': current_bar['Close'],
                    'ema': current_bar['ema']
                })
    
    return df_copy, pd.DataFrame(signals) if signals else pd.DataFrame()

# Keep the original function as a backup
def enhanced_ema_pullback_scanner(df, ema_period=21, min_pullback_candles=5):
    """Legacy function maintained for compatibility"""
    return detect_precise_ema_pullback(df, ema_period=ema_period)   

