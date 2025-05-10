import pandas as pd
import yfinance as yf
from .base_data_source import BaseDataSource
import datetime
from curl_cffi import requests
class YahooFinanceDataSource(BaseDataSource):
    """
    Data source that fetches historical data from Yahoo Finance using yfinance.
    Includes a method for searching symbols.
    """

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol and date range from Yahoo Finance.

        Args:
            symbol (str): The trading symbol (e.g., 'AAPL', 'MSFT', 'GOOG').
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A pandas DataFrame with historical data.
                          Includes 'Open', 'High', 'Low', 'Close', 'Volume'.
                          The index is a DatetimeIndex.
                          Returns an empty DataFrame if data fetching fails or no data is found.
        """
        try:
            # yfinance download handles date format and index automatically
            # auto_adjust=True adjusts OHLC prices for splits and dividends
            session = requests.Session(impersonate="chrome")
            data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, session=session)

            # --- Debugging prints for columns ---
            print(f"\n--- YahooFinanceDataSource Debug: Columns after yf.download for {symbol} ---")
            print(f"Type of columns: {type(data.columns)}")
            print(f"Columns: {data.columns}")
            if isinstance(data.columns, pd.MultiIndex):
                print(f"Number of levels in MultiIndex: {data.columns.nlevels}")
                print(f"Level names: {data.columns.names}")
                for level in range(data.columns.nlevels):
                    print(f"Level {level} values: {data.columns.get_level_values(level)}")
            print("----------------------------------------------------------------------\n")
            # --- End Debugging prints ---


            # --- Handle potential MultiIndex columns ---
            if isinstance(data.columns, pd.MultiIndex):
                # Assuming the structure is ('Price', 'Ticker') with Price as level 0
                # We want to keep only the 'Price' level names (level 0)
                try:
                    # Explicitly get level 0 values for the new column names
                    new_columns = data.columns.get_level_values(0)
                    data.columns = new_columns
                    print(f"Flattened MultiIndex columns for {symbol}.")
                except Exception as e:
                    print(f"Error flattening MultiIndex columns for {symbol}: {e}")
                    # If flattening fails, we might not be able to proceed
                    return pd.DataFrame()
            # --- End MultiIndex handling ---

            # --- Debugging prints for columns after flattening attempt ---
            print(f"\n--- YahooFinanceDataSource Debug: Columns after flattening attempt for {symbol} ---")
            print(f"Type of columns: {type(data.columns)}")
            print(f"Columns: {data.columns}")
            print("----------------------------------------------------------------------------\n")
            # --- End Debugging prints ---


            # After auto_adjust=True, the columns are typically 'Open', 'High', 'Low', 'Close', 'Volume'
            # The original 'Adj Close' is used to calculate the adjusted prices.
            # We will ensure the required columns are present.

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Use .reindex(columns=required_cols) to ensure order and handle potentially missing cols gracefully
            # This also helps if the flattening resulted in unexpected columns
            data = data.reindex(columns=required_cols)

            # Ensure index is DatetimeIndex (yfinance usually provides this, but check as safeguard)
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception as e:
                    print(f"Error converting index to DatetimeIndex for {symbol}: {e}")
                    return pd.DataFrame() # Return empty if index conversion fails


            # Drop rows with any NA values in required columns, as they are unusable for backtesting
            data.dropna(subset=required_cols, inplace=True)

            return data

        except Exception as e:
            print(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    def get_available_symbols(self) -> list[str]:
        """
        Returns an empty list as symbol search is handled by the search_symbols method.
        """
        return []

    def search_symbols(self, search_term: str) -> list[str]:
        """
        Searches for symbols using yfinance.Search based on the search term.

        Args:
            search_term (str): The term to search for (e.g., 'Apple', 'TSLA').

        Returns:
            list[str]: A list of matching symbols.
        """
        if not search_term:
            return [] # Return empty list if no search term

        try:
            # Use yfinance.Search to get search results
            session = requests.Session(impersonate="chrome")
            search_results = yf.Lookup(search_term, session=session).get_stock(count=10)
    # Extract symbols from the 'quotes' section of the results
            symbols = []

            if search_results is not None and 'exchange' in search_results.columns:
                for item in list(search_results['exchange'].keys()):
                    symbols.append(item)
                        
                    # Optionally, add the short name for better user experience
                    # if 'shortname' in item and item['shortname'] is not None:
                    #     symbols.append(f"{item['symbol']} - {item['shortname']}")

            # Return up to 20 unique symbols
            return list(dict.fromkeys(symbols))[:20] # Use dict.fromkeys to maintain order while getting unique

        except Exception as e:
            print(f"Error during yfinance search for '{search_term}': {e}")
            return [f"Error searching for {search_term}"] # Indicate error in the search box

