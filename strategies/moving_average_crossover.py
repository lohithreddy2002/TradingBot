import pandas as pd
from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """
    A simple Moving Average Crossover strategy.
    Generates a buy signal when the short-term MA crosses above the long-term MA,
    and a sell signal when the short-term MA crosses below the long-term MA.
    """

    def __init__(self, short_window: int = 40, long_window: int = 100):
        """
        Initializes the MovingAverageCrossover strategy.

        Args:
            short_window (int): The window size for the short-term moving average.
            long_window (int): The window size for the long-term moving average.
        """
        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window.")
        if short_window <= 0 or long_window <= 0:
             raise ValueError("Window sizes must be positive.")

        self._short_window = short_window
        self._long_window = long_window
        self._name = f"MA Crossover ({short_window}/{long_window})"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates buy and sell signals based on MA crossover.
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")

        signals = pd.Series(0, index=data.index)


        # Calculate moving averages
        short_mavg = data['Close'].rolling(window=self._short_window, min_periods=1).mean()
        long_mavg = data['Close'].rolling(window=self._long_window, min_periods=1).mean()




        # Generate signals
        # Buy signal when short MA crosses above long MA
        signals[short_mavg > long_mavg] = 1

        # Sell signal when short MA crosses below long MA
        signals[short_mavg < long_mavg] = -1



        # We only want signals on the day of the crossover, not every day the condition is met.
        # Take the difference to find the crossover points.
        # A change from -1 to 1 indicates a buy crossover (signal becomes 2)
        # A change from 1 to -1 indicates a sell crossover (signal becomes -2)
        # Other changes (0 to 1, 1 to 0, 0 to -1, -1 to 0) indicate entering/exiting the condition zone.
        signal_diff = signals.diff()

        # Reset signals and set only at crossover points
        signals[:] = 0
        signals[signal_diff == 2] = 1  # Buy signal on crossover up
        signals[signal_diff == -2] = -1 # Sell signal on crossover down

        # Ensure no consecutive signals of the same type (e.g., buy, buy)
        # This simplifies the backtesting logic assuming one position at a time.
        # Find the indices where signals are non-zero
        trade_indices = signals[signals != 0].index.tolist()

        # Keep only the first signal if consecutive signals of the same type appear
        if len(trade_indices) > 1:
            filtered_signals = pd.Series(0, index=data.index)
            last_signal = 0
            for idx in trade_indices:
                current_signal = signals.loc[idx]
                if current_signal != last_signal:
                    filtered_signals.loc[idx] = current_signal
                    last_signal = current_signal
            signals = filtered_signals


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
            "short_window": self._short_window,
            "long_window": self._long_window
        }
