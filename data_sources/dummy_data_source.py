import pandas as pd
import numpy as np
from .base_data_source import BaseDataSource
import datetime

class DummyDataSource(BaseDataSource):
    """
    A dummy data source that generates synthetic data for testing.
    """

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generates dummy historical data.
        """
        
        try:
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

        if start >= end:
            raise ValueError("Start date must be before end date.")

        date_range = pd.date_range(start=start, end=end, freq='D')
        n_days = len(date_range)

        if n_days == 0:
            return pd.DataFrame()

        # Generate synthetic price data
        # Start with a base price, add some trend and noise
        base_price = 100.0
        price_changes = np.random.randn(n_days) * 0.5
        price_changes[0] = 0 # Start with no change on the first day
        prices = base_price + np.cumsum(price_changes) + np.linspace(0, 10, n_days) # Add a slight upward trend

        # Ensure prices are positive
        prices = np.maximum(prices, 1.0)

        # Generate OHLC and Volume
        open_prices = prices + np.random.randn(n_days) * 0.2
        high_prices = np.maximum(open_prices, prices) + np.random.rand(n_days) * 0.5
        low_prices = np.minimum(open_prices, prices) - np.random.rand(n_days) * 0.5
        close_prices = prices

        # Ensure OHLC relationships are valid
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

        volume = np.random.randint(10000, 1000000, n_days)

        data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=date_range)

        # Add some missing data points randomly to simulate real-world data
        # missing_indices = np.random.choice(data.index, size=int(n_days * 0.05), replace=False)
        # data = data.drop(missing_indices)

        return data

    def get_available_symbols(self) -> list[str]:
        """
        Returns a list of dummy symbols.
        """
        return ["DUMMY_SYMBOL_1", "DUMMY_SYMBOL_2", "DUMMY_SYMBOL_3"]
