import pandas as pd
import datetime
import os
from .base_data_source import BaseDataSource
# Need to install alpaca-trade-api: pip install alpaca-trade-api
try:
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    class REST: # Dummy class if import fails
        def __init__(self, key_id, secret_key, base_url):
            print("Warning: alpaca-trade-api not installed. AlpacaDataSource will not work.")
            pass
        def get_bars(self, symbol, timeframe, start, end):
            print("AlpacaDataSource: get_bars called but alpaca-trade-api is not installed.")
            return []
        def search_assets(self, text, asset_class=None):
             print("AlpacaDataSource: search_assets called but alpaca-trade-api is not installed.")
             return []
    class TimeFrame: # Dummy class
        Day = '1D'


class AlpacaDataSource(BaseDataSource):
    """
    Data source that fetches historical data from Alpaca.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY to be set in Streamlit secrets.
    """

    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = 'https://paper-api.alpaca.markets'):
        """
        Initializes the AlpacaDataSource.

        Args:
            api_key (str): Your Alpaca API Key ID.
            secret_key (str): Your Alpaca Secret Key.
            base_url (str): The base URL for the Alpaca API (default is paper trading).
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("The 'alpaca-trade-api' library is not installed. Please install it (`pip install alpaca-trade-api`) to use AlpacaDataSource.")

        # Retrieve keys from Streamlit secrets if not provided
        if api_key is None:
             api_key = st.secrets.get("ALPACA_API_KEY")
        if secret_key is None:
             secret_key = st.secrets.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("Alpaca API Key ID and Secret Key must be provided or set in Streamlit secrets.")

        self.api = REST(api_key, secret_key, base_url)
        self._name = "Alpaca"


    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol and date range from Alpaca.

        Args:
            symbol (str): The trading symbol (e.g., 'AAPL', 'MSFT').
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A pandas DataFrame with historical data.
                          Includes 'Open', 'High', 'Low', 'Close', 'Volume'.
                          The index is a DatetimeIndex (timezone-aware).
                          Returns an empty DataFrame if data fetching fails or no data is found.
        """
        if not ALPACA_AVAILABLE:
            return pd.DataFrame()

        try:
            # Alpaca API uses ISO 8601 format for dates
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

            # Fetch daily bars
            bars = self.api.get_bars(symbol, TimeFrame.Day, start=start_dt, end=end_dt).df

            if bars.empty:
                return pd.DataFrame()

            # Ensure required columns are present and correctly named (Alpaca uses lowercase)
            bars.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            bars = bars.reindex(columns=required_cols)

            # Ensure index is DatetimeIndex (Alpaca's df index is usually timezone-aware DatetimeIndex)
            if not isinstance(bars.index, pd.DatetimeIndex):
                 try:
                      bars.index = pd.to_datetime(bars.index)
                 except Exception as e:
                      print(f"Error converting index to DatetimeIndex for {symbol} (Alpaca): {e}")
                      return pd.DataFrame()

            # Remove timezone information for consistency with other data sources if needed, or handle consistently
            # For simplicity here, we'll keep it as timezone-aware. Backtesting might need adjustment.
            # Example to remove timezone: bars.index = bars.index.tz_convert(None) # or .tz_localize(None)


            # Drop rows with any NA values in required columns
            bars.dropna(subset=required_cols, inplace=True)

            return bars

        except Exception as e:
            print(f"Error fetching data for {symbol} from Alpaca: {e}")
            return pd.DataFrame()

    def get_available_symbols(self) -> list[str]:
        """
        Alpaca API is primarily for trading specific assets.
        Searching is preferred over listing all available assets.
        Returns an empty list.
        """
        return []

    def search_symbols(self, search_term: str) -> list[str]:
        """
        Searches for assets using Alpaca API.

        Args:
            search_term (str): The term to search for (e.g., 'Apple', 'TSLA').

        Returns:
            list[str]: A list of matching symbols.
        """
        if not ALPACA_AVAILABLE:
            return ["Alpaca search not available (library not installed)"]

        if not search_term:
            return [] # Return empty list if no search term

        try:
            # Search for assets that are tradable
            assets = self.api.search_assets(text=search_term, asset_class='us_equity') # Assuming US equities
            # Filter for tradable assets
            tradable_symbols = [asset.symbol for asset in assets if asset.tradable]

            # Return up to 20 symbols
            return tradable_symbols[:20]

        except Exception as e:
            print(f"Error during Alpaca asset search for '{search_term}': {e}")
            return [f"Error searching for {search_term} (Alpaca)"]

    @property
    def name(self) -> str:
        """
        Returns the name of the data source.
        """
        return self._name

