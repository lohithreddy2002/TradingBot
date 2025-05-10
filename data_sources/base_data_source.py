import pandas as pd
from abc import ABC, abstractmethod

class BaseDataSource(ABC):
    """
    Abstract base class for data sources.
    New data sources should inherit from this class and implement the fetch_data method.
    """

    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol and date range.

        Args:
            symbol (str): The trading symbol (e.g., 'AAPL', 'BTC-USD').
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A pandas DataFrame with historical data.
                          Must include at least 'Open', 'High', 'Low', 'Close', 'Volume'.
                          The index should be a DatetimeIndex.
        """
        pass

    def get_available_symbols(self) -> list[str]:
        """
        Returns a list of available symbols for this data source.
        This can be implemented in subclasses if the data source has a predefined list.
        Otherwise, a default empty list is returned.
        """
        return []

    def search_symbols(self, search_term: str) -> list[str]:
        """
        Searches for symbols based on a search term.
        This can be implemented in subclasses if the data source provides a search API.
        Returns a list of matching symbols.
        """
        # Default implementation returns an empty list
        return []

