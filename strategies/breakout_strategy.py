import pandas as pd
import numpy as np # Import numpy
from .base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    """
    Breakout Trading Strategy (Manual Calculation).
    Generates a buy signal when the close price breaks above the highest high of the past N periods.
    Generates a sell signal when the close price breaks below the lowest low of the past N periods.
    """

    def __init__(self, lookback_period: int = 20):
        """
        Initializes the BreakoutStrategy.

        Args:
            lookback_period (int): The number of periods to look back for highest high and lowest low.
        """
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive.")

        self._lookback_period = lookback_period
        self._name = f"Breakout ({lookback_period})"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates buy and sell signals based on price breakouts.
        Breakout levels are calculated manually.
        """
        if 'Close' not in data.columns or 'High' not in data.columns or 'Low' not in data.columns:
            raise ValueError("Data must contain 'Close', 'High', and 'Low' columns.")

        signals = pd.Series(0, index=data.index)
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']

        # --- Manual Breakout Level Calculation ---
        # Calculate the highest high and lowest low over the lookback period
        # Use .rolling().max() and .rolling().min()
        highest_high = high_prices.rolling(window=self._lookback_period, min_periods=1).max()
        lowest_low = low_prices.rolling(window=self._lookback_period, min_periods=1).min()
        # --- End Manual Breakout Level Calculation ---


        # Generate buy signals: Close price breaks above the highest high
        # We look for the close price being greater than the highest high of the *previous* N periods
        # to avoid including the current day's high in the breakout level calculation.
        buy_signals = close_prices > highest_high.shift(1)

        # Generate sell signals: Close price breaks below the lowest low
        # We look for the close price being less than the lowest low of the *previous* N periods
        sell_signals = close_prices < lowest_low.shift(1)

        # Set signals in the Series
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        # Handle potential NaN values at the beginning due to the rolling window
        signals.fillna(0, inplace=True)

        # The backtester's consecutive signal filtering will be important here,
        # as this strategy can generate signals on many consecutive days.

        return signals

    @property
    def name(self) -> str:
        """
        Returns the name of the strategy.
        """
        return self._name

    def get_params(self) -> dict:
        """
        Returns the parameters of the strategy.
        """
        return {
            "lookback_period": self._lookback_period
        }
