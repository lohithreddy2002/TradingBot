import pandas as pd
import numpy as np # Import numpy
from .base_strategy import BaseStrategy

class MacdStrategy(BaseStrategy):
    """
    MACD Trading Strategy (Manual Calculation).
    Generates a buy signal when the MACD line crosses above the Signal line.
    Generates a sell signal when the MACD line crosses below the Signal line.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initializes the MacdStrategy.

        Args:
            fast_period (int): The period for the fast EMA.
            slow_period (int): The period for the slow EMA.
            signal_period (int): The period for the signal line EMA of the MACD.
        """
        if not (fast_period > 0 and slow_period > 0 and signal_period > 0):
             raise ValueError("Periods must be positive.")
        if fast_period >= slow_period:
             raise ValueError("Fast period must be smaller than slow period.")

        self._fast_period = fast_period
        self._slow_period = slow_period
        self._signal_period = signal_period
        self._name = f"MACD ({fast_period}, {slow_period}, {signal_period})"


    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates buy and sell signals based on MACD crossover of the Signal line.
        MACD is calculated manually.
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")

        signals = pd.Series(0, index=data.index)
        close_prices = data['Close']

        # --- Manual MACD Calculation ---
        # Calculate the 12-period EMA of the close price
        ema_fast = close_prices.ewm(span=self._fast_period, adjust=False).mean()

        # Calculate the 26-period EMA of the close price
        ema_slow = close_prices.ewm(span=self._slow_period, adjust=False).mean()

        # Calculate the MACD line
        macd_line = ema_fast - ema_slow

        # Calculate the 9-period EMA of the MACD line (the Signal line)
        signal_line = macd_line.ewm(span=self._signal_period, adjust=False).mean()
        # --- End Manual MACD Calculation ---


        # Generate buy signals: MACD line crosses above Signal line
        # We look for macd_line > signal_line where it was <= in the previous period
        buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

        # Generate sell signals: MACD line crosses below Signal line
        # We look for macd_line < signal_line where it was >= in the previous period
        sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        # Set signals in the Series
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        # Handle potential NaN values in signals due to initial calculation periods
        signals.fillna(0, inplace=True)

        # The backtester handles filtering for consecutive signals if needed.
        # This strategy inherently generates signals only at crossover points.

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
            "fast_period": self._fast_period,
            "slow_period": self._slow_period,
            "signal_period": self._signal_period
        }
