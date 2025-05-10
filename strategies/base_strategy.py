import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    New strategies should inherit from this class and implement the generate_signals method.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on historical data.

        Args:
            data (pd.DataFrame): The historical data (must include 'Close' price).

        Returns:
            pd.Series: A pandas Series with the same index as data, containing trading signals.
                       1 for buy signal, -1 for sell signal, 0 for no action.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the strategy.
        """
        pass

    def get_params(self) -> dict:
        """
        Returns a dictionary of parameters for the strategy.
        This can be implemented in subclasses if the strategy has configurable parameters.
        """
        return {}

