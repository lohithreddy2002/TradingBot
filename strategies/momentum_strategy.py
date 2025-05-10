import pandas as pd
import numpy as np # Import numpy
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """
    Momentum Trading Strategy (Manual Calculation).
    Generates a buy signal if the price change over a lookback period is positive.
    Generates a sell signal if the price change over a lookback period is negative.
    """

    def __init__(self, lookback_period: int = 10):
        """
        Initializes the MomentumStrategy.

        Args:
            lookback_period (int): The number of periods to look back for momentum calculation.
        """
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive.")

        self._lookback_period = lookback_period
        self._name = f"Momentum ({lookback_period})"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates buy and sell signals based on price momentum.
        Momentum is calculated manually.
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")

        signals = pd.Series(0, index=data.index)
        close_prices = data['Close']

        # --- Manual Momentum Calculation ---
        # Momentum is the current price minus the price N periods ago
        momentum = close_prices - close_prices.shift(self._lookback_period)
        # --- End Manual Momentum Calculation ---


        # Generate signals:
        # Buy if momentum is positive
        signals[momentum > 0] = 1
        # Sell if momentum is negative
        signals[momentum < 0] = -1

        # Handle potential NaN values at the beginning due to the shift
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
