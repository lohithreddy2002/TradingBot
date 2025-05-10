import pandas as pd
import datetime
import os
from .base_data_source import BaseDataSource
# Need to install kiteconnect: pip install kiteconnect
try:
    from kiteconnect import KiteConnect
    ZERODHA_AVAILABLE = True
except ImportError:
    ZERODHA_AVAILABLE = False
    class KiteConnect: # Dummy class if import fails
        def __init__(self, api_key):
            print("Warning: kiteconnect not installed. ZerodhaDataSource will not work.")
            pass
        def generate_session(self, request_token, api_secret):
             print("ZerodhaDataSource: generate_session called but kiteconnect is not installed.")
             return {"access_token": "dummy_token"}
        def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False):
             print("ZerodhaDataSource: historical_data called but kiteconnect is not installed.")
             return []
        def instruments(self, exchange=None):
             print("ZerodhaDataSource: instruments called but kiteconnect is not installed.")
             return []


class ZerodhaDataSource(BaseDataSource):
    """
    Data source that fetches historical data from Zerodha Kite.
    Requires ZERODHA_API_KEY and ZERODHA_API_SECRET to be set in Streamlit secrets.
    Authentication requires a manual step to get the request token.
    This implementation assumes the session is already generated and access token is available.
    A more robust implementation would handle the full OAuth flow.
    For simplicity, this version will rely on the access token being set in secrets
    or a manual input mechanism (which is outside the scope of this file).
    """

    def __init__(self, api_key: str = None, access_token: str = None):
        """
        Initializes the ZerodhaDataSource.

        Args:
            api_key (str): Your Zerodha Kite API Key.
            access_token (str): Your Zerodha Kite Access Token.
        """
        if not ZERODHA_AVAILABLE:
            raise ImportError("The 'kiteconnect' library is not installed. Please install it (`pip install kiteconnect`) to use ZerodhaDataSource.")

        # Retrieve keys from Streamlit secrets if not provided
        if api_key is None:
             api_key = st.secrets.get("ZERODHA_API_KEY")
        if access_token is None:
             # In a real app, the access token would be obtained via OAuth flow
             # For this simplified version, we assume it's in secrets for demonstration
             access_token = st.secrets.get("ZERODHA_ACCESS_TOKEN")


        if not api_key or not access_token:
            raise ValueError("Zerodha API Key and Access Token must be provided or set in Streamlit secrets.")

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._name = "Zerodha"
        self._instruments_cache = None # Cache for instruments list


    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given instrument token from Zerodha Kite.
        Note: Zerodha API typically uses instrument tokens, not symbols directly for historical data.
        This method expects the 'symbol' argument to be the instrument token string.

        Args:
            symbol (str): The Zerodha instrument token as a string.
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A pandas DataFrame with historical data.
                          Includes 'Open', 'High', 'Low', 'Close', 'Volume'.
                          The index is a DatetimeIndex.
                          Returns an empty DataFrame if data fetching fails or no data is found.
        """
        if not ZERODHA_AVAILABLE:
            return pd.DataFrame()

        try:
            # Zerodha API requires instrument token as integer
            instrument_token = int(symbol) # Assume symbol is the instrument token

            # Fetch historical data (daily interval)
            # The API expects datetime objects or strings in 'YYYY-MM-DD' format
            from_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            to_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

            # Using 'day' interval for historical data
            historical_data = self.kite.historical_data(instrument_token, from_date, to_date, 'day')

            if not historical_data:
                return pd.DataFrame()

            # Convert list of dictionaries to DataFrame
            data = pd.DataFrame(historical_data)

            # Rename columns to match expected format
            data.rename(columns={
                'date': 'Timestamp', # Zerodha uses 'date'
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            # Set 'Timestamp' as index and convert to DatetimeIndex
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data.set_index('Timestamp', inplace=True)

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data.reindex(columns=required_cols)


            # Ensure index is DatetimeIndex and sorted
            if not isinstance(data.index, pd.DatetimeIndex):
                 try:
                      data.index = pd.to_datetime(data.index)
                 except Exception as e:
                      print(f"Error converting index to DatetimeIndex for token {symbol} (Zerodha): {e}")
                      return pd.DataFrame()

            data.sort_index(inplace=True)

            # Drop rows with any NA values in required columns
            data.dropna(subset=required_cols, inplace=True)


            return data

        except ValueError:
             print(f"Error: Invalid instrument token provided for Zerodha: {symbol}. Must be an integer.")
             return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for token {symbol} from Zerodha: {e}")
            return pd.DataFrame()


    def get_available_symbols(self) -> list[str]:
        """
        Fetches a list of all tradable instruments from Zerodha.
        This can be a large list.
        """
        if not ZERODHA_AVAILABLE:
            return ["Zerodha symbols not available (library not installed)"]

        if self._instruments_cache is not None:
             return self._instruments_cache # Return cached list

        try:
            # Fetch instruments list (can be slow)
            instruments = self.kite.instruments()

            # Extract symbols (or instrument tokens) - let's return symbols for searchability
            # A real application might need to map symbols to instrument tokens for fetching data
            symbols = [instr['tradingsymbol'] for instr in instruments]

            self._instruments_cache = symbols # Cache the list
            return symbols

        except Exception as e:
            print(f"Error fetching instruments from Zerodha: {e}")
            return [f"Error fetching Zerodha symbols: {e}"]


    def search_symbols(self, search_term: str) -> list[str]:
        """
        Searches for instruments based on the search term.
        Fetches the full list and filters.

        Args:
            search_term (str): The term to search for (e.g., 'RELIANCE', 'INFY').

        Returns:
            list[str]: A list of matching trading symbols.
        """
        if not ZERODHA_AVAILABLE:
            return ["Zerodha search not available (library not installed)"]

        if not search_term:
            return [] # Return empty list if no search term

        try:
            # Get the full list of instruments (from cache or fetch)
            all_symbols = self.get_available_symbols()

            if "Error fetching Zerodha symbols" in all_symbols:
                 return all_symbols # Return the error message

            # Filter the list based on the search term (case-insensitive)
            search_term_lower = search_term.lower()
            matching_symbols = [symbol for symbol in all_symbols if search_term_lower in symbol.lower()]

            # Return up to 20 matching symbols
            return matching_symbols[:20]

        except Exception as e:
            print(f"Error during Zerodha symbol search for '{search_term}': {e}")
            return [f"Error searching for {search_term} (Zerodha)"]


    @property
    def name(self) -> str:
        """
        Returns the name of the data source.
        """
        return self._name

