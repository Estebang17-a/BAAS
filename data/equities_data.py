import yfinance as yf
import pandas as pd

# Function to get tickers of S&P 500 companies (based on a Wikipedia table)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    symbols = sp500_table['Symbol'].tolist()
    return symbols

# Download OHLC data for each ticker and store it in a dictionary
def download_equities_ohlc_data(timeframe='1d', symbols=None, period='1y', save_path=None):

    if symbols is None:
        symbols = get_sp500_tickers()

    try:
        # Downloading data in one call
        ohlc_data = yf.download(symbols, period=period, interval=timeframe, group_by='ticker', threads=True)
        print("Data downloaded successfully.")

        # Saving data to CSV
        if save_path:
            ohlc_data.to_csv(f"{save_path}.csv")
            
        return ohlc_data

    except Exception as e:
        print(f"Failed to download data: {e}")
        return None

# Now, `ohlc_data` is a dictionary containing the OHLC data for each ticker
# You can access it like this: ohlc_data['AAPL']['Close'] to get close data for Apple Inc., for instance

if __name__ == '__main__':
    data = download_equities_ohlc_data()
    print(data)