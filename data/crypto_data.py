import pandas as pd
import ccxt
from datetime import timedelta

def download_crypto_ohlc_data(exchange='bybit', symbols=None, timeframe='1d', since=None, periods=365):
    # Initialize exchange
    print(f"Initializing data download for: {exchange}, Timeframe: {timeframe}, Periods: {periods}...")
    exchange = getattr(ccxt, exchange)({'enableRateLimit': True})

    # CCXT periods limit is 1000
    if periods > 1000:
        periods = 1000

    # Fetch symbols if not provided
    if symbols is None:
        markets = exchange.load_markets()
        #symbols = [market for market in markets if market.endswith('USDT')] # to delete
        symbols = [market for market in markets if 'USDT:USDT' in market]
        print(f"Downloading data for {len(symbols)} symbols")

    ohlc_data = {}
    for i, symbol in enumerate(symbols):
        if(symbol[-1] == 'C' or symbol[-1] == 'P'):
            continue
        try:

            data = exchange.fetch_ohlcv(symbol, timeframe, since, periods)
            if data:
                df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                ohlc_data[symbol] = df

                # Print for the first and last symbol
                if i == 0 or i == len(symbols) - 1:
                    print(f"Data for {symbol}:")
                    print(f"  Start: {df['Timestamp'].iloc[0]}")
                    print(f"  End: {df['Timestamp'].iloc[-1]}")
                    print(f'Last row: \n{df.iloc[-1]}')
                    # Check the interval between data points
                    interval = df['Timestamp'].iloc[1] - df['Timestamp'].iloc[0]
                    print(f"\n  Expected Interval: {timeframe}, Actual Interval: {interval}")

        except Exception as e:
            print(f"Failed to download data for {symbol}: {e}")

    # Combine all data into a single DataFrame
    combined_df = pd.concat(ohlc_data, axis=1)

    # Save to CSV
    combined_df.to_csv("all_perpetuals_ohlc_data.csv")

    print(f'Finished downloading data for {len(ohlc_data)} symbols')

    return combined_df

if __name__ == '__main__':
    download_crypto_ohlc_data()
