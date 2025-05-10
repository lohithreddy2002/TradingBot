import pandas as pd
import requests
import datetime
import os
from .base_data_source import BaseDataSource

class AlphaVantageDataSource(BaseDataSource):
    """
    Data source for fetching historical data from Alpha Vantage.
    Requires an Alpha Vantage API key set as an environment variable ALPHA_VANTAGE_API_KEY.
    """

    def __init__(self):
        """
        Initializes the AlphaVantageDataSource.
        Loads the API key from environment variables.
        """
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set.")
        self.base_url = "https://www.alphavantage.co/query"
        self._name = "Alpha Vantage"

   

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical daily data for a given symbol and date range from Alpha Vantage.

        Args:
            symbol (str): The trading symbol (e.g., 'AAPL').
            start_date (str): The start date for the data (YYYY-MM-DD).
            end_date (str): The end date for the data (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A pandas DataFrame with historical data.
                          Includes 'Open', 'High', 'Low', 'Close', 'Volume'.
                          The index is a DatetimeIndex.
            Raises:
                ValueError: If API key is missing or invalid date format.
                requests.exceptions.RequestException: For network or API errors.
                KeyError: If the expected data structure is not found in the response.
        """
        if not self.api_key:
             raise ValueError("API key is not set for Alpha Vantage.")

        try:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

        # Alpha Vantage 'full' output size gives up to 20 years of history
        # We will fetch the full data and then filter by date range.
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full", # Use 'full' to get historical data
            "apikey": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()

            # Check for API errors in the response
            if "Error Message" in data:
                raise requests.exceptions.RequestException(f"Alpha Vantage API Error: {data['Error Message']}")
            if "Note" in data:
                print(f"Alpha Vantage API Note: {data['Note']}") # Display notes as warnings (e.g., rate limits)


            # The time series data is nested under "Time Series (Daily)"
            time_series_data = data.get("Time Series (Daily)")
            if not time_series_data:
                 # If no time series data, it might be an invalid symbol or no data available
                 if "Information" in data:
                      raise KeyError(f"Alpha Vantage API: {data['Information']}") # Often indicates invalid symbol
                 else:
                      raise KeyError("Could not find 'Time Series (Daily)' in Alpha Vantage response.")


            # Convert the nested dictionary to a pandas DataFrame
            # The keys are dates (strings), and values are dictionaries of OHLCV data (strings)
            df = pd.DataFrame.from_dict(time_series_data, orient='index')

            # Rename columns and convert to appropriate types
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype({
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': int
            })

            # Convert index to DatetimeIndex
            df.index = pd.to_datetime(df.index)

            # Sort the index in ascending order (oldest to newest)
            df.sort_index(inplace=True)

            # Filter by the requested date range
            df = df[(df.index >= start) & (df.index <= end)]

            return df

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Network or API error fetching data from Alpha Vantage: {e}")
        except KeyError as e:
             raise KeyError(f"Error parsing Alpha Vantage response: {e}. Response structure may have changed or symbol is invalid.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while processing Alpha Vantage data: {e}")

    def get_available_symbols(self) -> list[str]:
        """
        Returns an empty list as symbol search is handled by the search_symbols method.
        """
        return []
    
    def search_symbols(self, search_term: str) -> list[str]:
        """
        Searches for symbols using the Alpha Vantage SYMBOL_SEARCH endpoint.

        Args:
            search_term (str): The search term (e.g., 'Apple', 'AAPL').

        Returns:
            list[str]: A list of matching symbols.
        """
        if not self.api_key:
             print("Warning: API key is not set for Alpha Vantage, cannot perform symbol search.")
             return []

        if not search_term:
             return [] # Don't search with an empty term

        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": search_term,
            "apikey": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check for API errors or no results
            if "Error Message" in data:
                 print(f"Alpha Vantage API Error during search: {data['Error Message']}")
                 return []
            if "bestMatches" not in data:
                 # No matches found
                 return []

            # Extract symbols from the bestMatches list
            symbols = [match['1. symbol'] for match in data['bestMatches']]
            return symbols

        except requests.exceptions.RequestException as e:
            print(f"Network or API error during Alpha Vantage symbol search: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during Alpha Vantage symbol search: {e}")
            return []

    # Alpha Vantage does not have a simple endpoint to list all symbols
    # We will rely on the search_symbols method for symbol selection in the UI.
    # get_available_symbols will return an empty list as per the base class default.
    # def get_available_symbols(self) -> list[str]:
    #     return []

