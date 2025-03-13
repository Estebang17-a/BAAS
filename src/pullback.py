import pandas as pd
import numpy as np

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def detect_ema_pullback(df, ema_period=21, pullback_threshold=0.3):
    """
    Detect EMA pullback pattern for trading 
    Parameters:
    df: DataFrame with OHLC data (must have 'open', 'high', 'low', 'close' columns)
    ema_period: Period for EMA calculation
    trend_lookback: Bars to look back to establish trend
    pullback_threshold: How deep pullback should be (0-1.0)
    """
    df_copy = df.copy()

    # Calculate EMA
    df_copy['ema'] = calculate_ema(df_copy['Close'], ema_period)
    df_copy['ema_2'] = calculate_ema(df_copy['Close'], ema_period * 2) # Additional ema for trend confirmation

    # Identify trend using EMA slope
    df_copy['ema_slope'] = df_copy['ema'].diff(5)
    df_copy['ema_2_slope'] = df_copy['ema_2'].diff(10)  

    # Determine if it's in uptrend or downtrend
    df_copy['uptrend'] = (df_copy['ema'] > df_copy['ema_2']) & (df_copy['ema_slope'] > 0)
    df_copy['downtrend'] = (df_copy['ema'] < df_copy['ema_2']) & (df_copy['ema_slope'] < 0)

    # Calculate distance from EMa
    df_copy['distance_from_ema'] = (df_copy['Close'] - df_copy['ema']) / df_copy['ema']

    # Find potential long entries - pullbacks to EMA in uptrend
    df_copy['pullback_to_ema_in_uptrend'] = (
        df_copy['uptrend'] &
        (df_copy['distance_from_ema'] <= pullback_threshold) &
        (df_copy['distance_from_ema'] >= -pullback_threshold)
    )

    # Find potential short entries - pullbacks to EMA in downtrend
    df_copy['pullback_to_ema_in_downtrend'] = (
        df_copy['downtrend'] & 
        (df_copy['distance_from_ema'] >= -pullback_threshold) &
        (df_copy['distance_from_ema'] <= pullback_threshold)
    )

    return df_copy 

def enhanced_ema_pullback_scanner(df, ema_period=21, min_pullback_candles=3):
    """
    Advanced detector
    """

    df_copy = df.copy()

    # Calculate EMA
    df_copy['ema'] = calculate_ema(df_copy['Close'], ema_period)
    df_copy['ema_2'] = calculate_ema(df_copy['Close'], ema_period * 2) # Additional ema for trend confirmation

    # Determine trend
    df_copy['trend_direction'] = np.where(df_copy['ema'] > df_copy['ema_2'], 1, -1)

    # Calculate slope for the trend strength
    df_copy['ema_slope'] = df_copy['ema'].pct_change(5) * 100

    # Identify strong trends
    df_copy['strong_uptrend'] = (df_copy['trend_direction'] == 1) & (df_copy['ema_slope'] > 0.6)
    df_copy['strong_downtrend'] = (df_copy['trend_direction'] == -1) & (df_copy['ema_slope'] < -0.6)

    # Find pullback patterns
    signals = []

    for i in range(min_pullback_candles + 10, len(df_copy)):
        window = df_copy.iloc[i-min_pullback_candles-10:i+1]

        # For long setup
        if window['strong_uptrend'].iloc[-1]:
            # Check if price pulled back to EMA
            pullback_selection = window.iloc[-min_pullback_candles-5:-1]    

            # Price touched or crossed EMa during pullback
            touched_ema = any(
                (pullback_selection['Low'] <= pullback_selection['ema']) &
                (pullback_selection['high'] >= pullback_selection['ema'])
            )

            # Check for breakout candle
            breakout = (
                window['Close'].iloc[-1] > window['High'].iloc[-2] and
                window['Close'].iloc[-1] > window['ema'].iloc[-1] and
                window['Close'].iloc[-1] > window['Open'].iloc[-1]  # Bullish candle
            )

            if touched_ema and breakout:
                signals.append({
                    'index': window.index[-1],
                    'type': 'long',
                    'price': window['Close'].iloc[-1],
                    'ema': window['ema'].iloc[-1]
                })

        # For short setup
        elif window['strong_downtrend'].iloc[-1]:
            # Check if price pulled back to EMA
            pullback_selection = window.iloc[-min_pullback_candles-5:-1]

            # Price touched or crossed EMA during pullback
            touched_ema = any(
                (pullback_selection['High'] >= pullback_selection['ema']) &
                (pullback_selection['Low'] <= pullback_selection['ema'])
            )

            # Check for breakdown candle
            breakdown = (
                window['Close'].iloc[-1] < window['Low'].iloc[-2] and
                window['Close'].iloc[-1] < window['ema'].iloc[-1] and
                window['Close'].iloc[-1] < window['Open'].iloc[-1]  # Bearish candle
            )
            
            if touched_ema and breakdown:
                signals.append({
                    'index': window.index[-1],
                    'type': 'short',
                    'price': window['Close'].iloc[-1],
                    'ema': window['ema'].iloc[-1]
                })  

    return df_copy, pd.DataFrame(signals) if signals else pd.DataFrame()   

