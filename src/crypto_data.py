import pandas as pd
import ccxt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
import pytz

def fetch_ohlcv(exchange, symbol, timeframe, since, periods):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, since, periods)
        if data:
            df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
            return symbol, df
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")
    return symbol, None

def download_crypto_ohlc_data(exchange='bybit', symbols=None, timeframe='1d', since=None, periods=150, save_csv=False):
    initial_time = time.time()
    print(f"Initializing data download for: {exchange}, Timeframe: {timeframe}, Periods: {periods}...\n")
    exchange = getattr(ccxt, exchange)({'enableRateLimit': True})

    if periods > 1000:
        periods = 1000

    # Calculate the end time as the start of the current hour
    end_time = datetime.now(pytz.utc).replace(minute=0, second=0, microsecond=0)
    
    # Calculate the start time based on the number of periods
    if timeframe == '1h':
        start_time = end_time - timedelta(hours=periods)
    elif timeframe == '1d':
        start_time = end_time - timedelta(days=periods)
    elif timeframe == '1m':
        start_time = end_time - timedelta(minutes=periods)
    elif timeframe == '5m':
        start_time = end_time - timedelta(minutes=periods * 5)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    since = int(start_time.timestamp() * 1000)

    if symbols is None:
        markets = exchange.load_markets()
        symbols = [market for market in markets if market.endswith('USDT')]
        print(f"Downloading data for {len(symbols)} symbols")

    ohlc_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_ohlcv, exchange, symbol, timeframe, since, periods): symbol for symbol in symbols if symbol[-1] not in ['C', 'P']}
        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                symbol, df = future.result()
                if df is not None:
                    ohlc_data[symbol] = df
                    if i == 0 or i == len(symbols) - 1:
                        print(f"Data for {symbol}:")
                        print(f"  Start: {df['Timestamp'].iloc[0]}")
                        print(f"  End: {df['Timestamp'].iloc[-1]}")
                        print(f'Last row: \n{df.iloc[-1]}')
                        interval = df['Timestamp'].iloc[1] - df['Timestamp'].iloc[0]
                        print(f"\n  Expected Interval: {timeframe}, Actual Interval: {interval}")

            except Exception as e:
                print(f"Failed to download data for {symbol}: {e}")

    combined_df = pd.concat(ohlc_data, axis=1)

    if save_csv:
        combined_df.to_csv("all_perpetuals_ohlc_data.csv")

    print(f'Finished downloading data for {len(ohlc_data)} symbols in {time.time() - initial_time:.2f} seconds.')

    return combined_df

if __name__ == '__main__':
    download_crypto_ohlc_data()