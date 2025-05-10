import pandas as pd
import numpy as np # Import numpy
from .base_strategy import BaseStrategy

class RsiStrategy(BaseStrategy):
    """
    RSI-Based Trading Strategy (Manual Calculation).
    Generates a buy signal when RSI crosses below a lower threshold and then back above it.
    Generates a sell signal when RSI crosses above an upper threshold and then back below it.
    """

    def __init__(self, rsi_length: int = 14, rsi_buy_threshold: int = 30, rsi_sell_threshold: int = 70):
        """
        Initializes the RsiStrategy.

        Args:
            rsi_length (int): The lookback period for the RSI calculation.
            rsi_buy_threshold (int): The lower RSI threshold for buy signals.
            rsi_sell_threshold (int): The upper RSI threshold for sell signals.
        """
        if rsi_length <= 1: # RSI requires at least 2 periods for the first calculation
            raise ValueError("RSI length must be greater than 1.")
        if rsi_buy_threshold >= rsi_sell_threshold:
             raise ValueError("RSI buy threshold must be less than sell threshold.")
        if not (0 <= rsi_buy_threshold <= 100) or not (0 <= rsi_sell_threshold <= 100):
             raise ValueError("RSI thresholds must be between 0 and 100.")

        self._rsi_length = rsi_length
        self._rsi_buy_threshold = rsi_buy_threshold
        self._rsi_sell_threshold = rsi_sell_threshold
        self._name = f"RSI ({rsi_length}, Buy: {rsi_buy_threshold}, Sell: {rsi_sell_threshold})"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates buy and sell signals based on RSI crossovers of thresholds.
        RSI is calculated manually.
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")

        signals = pd.Series(0, index=data.index)
        close_prices = data['Close']

        # --- Manual RSI Calculation ---
        # Calculate price changes
        delta = close_prices.diff()

        # Get gains (up) and losses (down)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate the average gain and average loss using Exponential Moving Average (EMA)
        # This is the standard way RSI is calculated.
        avg_gain = gain.ewm(com=self._rsi_length - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self._rsi_length - 1, adjust=False).mean()

        # Calculate Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Calculate Relative Strength Index (RSI)
        # Handle division by zero if avg_loss is 0
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100 # If avg_loss is 0, RS is infinite, RSI is 100
        # --- End Manual RSI Calculation ---


        # Generate potential buy signals: RSI crosses below buy threshold
        buy_condition_below = rsi < self._rsi_buy_threshold
        # Generate potential sell signals: RSI crosses above sell threshold
        sell_condition_above = rsi > self._rsi_sell_threshold

        # Generate actual buy signals: Cross below threshold and then back above in the next period
        # We look for a transition from buy_condition_below being True to False
        # This is a common way to signal entry after oversold condition reverses
        buy_signals = (buy_condition_below.shift(1) == True) & (buy_condition_below == False) & (rsi > self._rsi_buy_threshold)

        # Generate actual sell signals: Cross above threshold and then back below in the next period
        # We look for a transition from sell_condition_above being True to False
        # This signals exit after overbought condition reverses
        sell_signals = (sell_condition_above.shift(1) == True) & (sell_condition_above == False) & (rsi < self._rsi_sell_threshold)


        # Set signals in the Series
        signals[buy_signals] = 1
        signals[sell_signals] = -1

        # Handle potential NaN values in signals due to initial RSI calculation period
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
            "rsi_length": self._rsi_length,
            "rsi_buy_threshold": self._rsi_buy_threshold,
            "rsi_sell_threshold": self._rsi_sell_threshold
        }
