import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.base_strategy import BaseStrategy
from utils.plotting import plot_equity_curve # plot_price_with_signals might need adaptation for multi-asset
import datetime

class Backtester:
    """
    Core backtesting engine for single or multi-asset backtesting.
    Simulates trades based on strategy signals across multiple assets
    and calculates portfolio-level performance metrics, including Sharpe,
    Sortino, and Calmar ratios, with optional global stop loss and take profit.
    Includes position sizing methods.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initializes the Backtester.

        Args:
            initial_capital (float): The starting capital for the backtest.
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        self.initial_capital = initial_capital
        self.portfolio = None # Will be a DataFrame tracking cash and total equity
        self.asset_holdings = {} # Dictionary to track holdings per asset {symbol: pd.Series}
        self.trades = [] # List to store trade details
        self._risk_free_rate = 0.0 # Assume 0% risk-free rate for simplicity

    def run_backtest(self, data: dict[str, pd.DataFrame], strategy: BaseStrategy,
                     stop_loss_pct: float = 0, take_profit_pct: float = 0,
                     position_sizing_method: str = 'Full Equity', position_sizing_value: float = 1.0,
                     initial_allocation: dict[str, float] = None): # Added initial_allocation parameter
        """
        Runs the backtest simulation for one or more assets.

        Args:
            data (dict[str, pd.DataFrame]): A dictionary where keys are symbols
                                            and values are pandas DataFrames with historical data.
                                            Each DataFrame must include 'Open', 'High', 'Low', 'Close', 'Volume'.
                                            Indices must be DatetimeIndex and ideally aligned or overlapping.
            strategy (BaseStrategy): The trading strategy to backtest.
            stop_loss_pct (float): Percentage for global stop loss (e.g., 0.05 for 5%). 0 means no stop loss.
                                   Applied per-trade.
            take_profit_pct (float): Percentage for global take profit (e.g., 0.10 for 10%). 0 means no take profit.
                                     Applied per-trade.
            position_sizing_method (str): Method for determining position size ('Full Equity', 'Fixed Amount', 'Percentage of Equity').
            position_sizing_value (float): Value used by the position sizing method (e.g., dollar amount or percentage).
            initial_allocation (dict[str, float], optional): A dictionary specifying the initial percentage
                                                             allocation per symbol {symbol: percentage}.
                                                             NOTE: The backtester currently does NOT use this
                                                             to set up initial positions; it starts with 100% cash.
                                                             This parameter is included to match the UI and for
                                                             future enhancement of the backtester logic. Defaults to None.

        Returns:
            dict: A dictionary containing backtest results and portfolio-level performance metrics.
        """
        if not data:
            return {"error": "No data provided for backtesting."}

        # --- Data Validation and Preparation ---
        # Find the common date range across all assets
        all_indices = [df.index for df in data.values()]
        if not all_indices:
             return {"error": "No dataframes provided in the data dictionary."}

        # Find the intersection of all indices to get the dates where data exists for all symbols
        common_index = all_indices[0]
        for index in all_indices[1:]:
            common_index = common_index.intersection(index)

        if common_index.empty:
             return {"error": "No common dates found across all selected assets. Cannot backtest."}

        # Reindex all dataframes to the common index and fill missing values (e.g., with previous day's close or NaN)
        # Using forward fill (ffill) for OHLCV data on missing dates
        aligned_data = {}
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                 try:
                     df.index = pd.to_datetime(df.index)
                 except Exception as e:
                     return {"error": f"Failed to convert index to DatetimeIndex for {symbol}: {e}"}

            # Ensure required columns are present before reindexing
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = pd.NA # Add missing columns with NA

            # Reindex and forward fill
            aligned_df = df.reindex(common_index).fillna(method='ffill')
            # After ffill, there might still be NaNs at the very beginning if the first common date has no data
            aligned_df.dropna(subset=['Close'], inplace=True) # Drop rows where Close is still NaN

            if aligned_df.empty:
                 print(f"Warning: No valid data for {symbol} after alignment and cleaning. Skipping.")
                 continue # Skip this symbol if no valid data remains

            aligned_data[symbol] = aligned_df[required_cols] # Store only required columns

        if not aligned_data:
             return {"error": "No valid data remaining for any asset after alignment and cleaning."}

        # Update common_index based on the indices of aligned_data
        common_index = list(aligned_data.values())[0].index # Use the index of the first aligned dataframe


        # --- Generate Signals for Each Asset ---
        signals = {}
        for symbol, df in aligned_data.items():
            try:
                asset_signals = strategy.generate_signals(df.copy()) # Generate signals for each asset
                if not isinstance(asset_signals.index, pd.DatetimeIndex):
                     return {"error": f"Signals index for {symbol} is not a DatetimeIndex."}
                # Reindex signals to the common index, filling missing with 0
                signals[symbol] = asset_signals.reindex(common_index, fill_value=0)
            except Exception as e:
                print(f"Error generating signals for {symbol} with strategy {strategy.name}: {e}. Skipping asset.")
                # If signal generation fails for an asset, treat its signals as all 0
                signals[symbol] = pd.Series(0, index=common_index)


        # Check if any signals were generated across all assets
        total_signals = sum([s.abs().sum() for s in signals.values()])
        if total_signals == 0:
             return {"error": "Strategy generated no signals across all assets."}


        # --- Initialize Portfolio and Asset Holdings ---
        self.portfolio = pd.DataFrame(index=common_index)
        self.portfolio['cash'] = self.initial_capital
        self.portfolio['total'] = self.initial_capital # Total portfolio value

        self.asset_holdings = {symbol: pd.Series(0.0, index=common_index) for symbol in aligned_data.keys()}

        # Simulation state variables per asset
        asset_states = {symbol: {'position': 0, 'entry_price': 0.0, 'stop_loss_price': None, 'take_profit_price': None}
                        for symbol in aligned_data.keys()}

        trade_count = 0

        # --- Simulate Trades Day by Day ---
        print("\n--- Multi-Asset Backtesting Loop ---")
        print(f"Backtesting over {len(common_index)} days for {list(aligned_data.keys())}")
        print(f"Position Sizing Method: {position_sizing_method}, Value: {position_sizing_value}")
        print(f"Stop Loss %: {stop_loss_pct}, Take Profit %: {take_profit_pct}")
        # Print initial allocation if provided (for debugging/verification)
        if initial_allocation:
             print(f"Initial Allocation (provided): {initial_allocation}")
        print("------------------------------------\n")


        try:
            for i in range(len(common_index)):
                current_index_label = common_index[i]
                previous_index_label = common_index[i-1] if i > 0 else None

                # Carry over previous day's portfolio state
                if previous_index_label is not None:
                    self.portfolio.loc[current_index_label, 'cash'] = self.portfolio.loc[previous_index_label, 'cash']
                    for symbol in aligned_data.keys():
                         self.asset_holdings[symbol].loc[current_index_label] = self.asset_holdings[symbol].loc[previous_index_label]
                # else: Initial state is already set

                # --- Process each asset for the current day ---
                for symbol, df in aligned_data.items():
                    if current_index_label not in df.index:
                        # This should ideally not happen after reindexing to common_index, but as a safeguard
                        print(f"Warning: Data missing for {symbol} on {current_index_label}. Skipping processing for this asset on this day.")
                        continue

                    current_open = df['Open'].loc[current_index_label]
                    current_high = df['High'].loc[current_index_label]
                    current_low = df['Low'].loc[current_index_label]
                    current_close = df['Close'].loc[current_index_label]
                    current_signal = signals[symbol].loc[current_index_label]

                    asset_state = asset_states[symbol]
                    position = asset_state['position']
                    entry_price = asset_state['entry_price']
                    stop_loss_price = asset_state['stop_loss_price']
                    take_profit_price = asset_state['take_profit_price']
                    shares_held = self.asset_holdings[symbol].loc[current_index_label] # Get current holdings for this asset

                    # --- Handle Stop Loss and Take Profit (Check during the day) ---
                    # Check for exits only if in a long position
                    if position == 1:
                        exit_price = None
                        exit_type = None

                        # Check Take Profit first (if TP is set)
                        if take_profit_pct > 0 and take_profit_price is not None:
                            # Check if high of the day hit or exceeded TP price
                            if current_high >= take_profit_price:
                                exit_price = take_profit_price # Assume execution at TP price
                                exit_type = 'Take Profit'
                                # print(f"{symbol} TP hit at {current_index_label}: {exit_price}")

                        # Check Stop Loss (if SL is set and TP was not hit)
                        if exit_price is None and stop_loss_pct > 0 and stop_loss_price is not None:
                            # Check if low of the day hit or fell below SL price
                            if current_low <= stop_loss_price:
                                exit_price = stop_loss_price # Assume execution at SL price
                                exit_type = 'Stop Loss'
                                # print(f"{symbol} SL hit at {current_index_label}: {exit_price}")

                        # Execute exit if SL or TP was hit
                        if exit_price is not None and shares_held > 0:
                            revenue = shares_held * exit_price
                            profit_loss = (exit_price - entry_price) * shares_held
                            self.portfolio.loc[current_index_label, 'cash'] += revenue # Add cash to portfolio
                            self.asset_holdings[symbol].loc[current_index_label] = 0.0 # Zero out holdings for this asset
                            asset_state['position'] = 0
                            asset_state['entry_price'] = 0.0
                            asset_state['stop_loss_price'] = None # Reset SL/TP after exiting
                            asset_state['take_profit_price'] = None
                            self.trades.append({
                                'symbol': symbol,
                                'type': 'Sell',
                                'date': current_index_label,
                                'price': exit_price,
                                'shares': shares_held, # Record shares sold
                                'revenue': revenue,
                                'profit_loss': profit_loss,
                                'exit_type': exit_type # Record how the position was exited
                            })
                            shares_held = 0.0 # Update local shares_held after selling


                    # --- Handle Trading Signals (Only if no exit occurred and currently flat for this asset) ---
                    if position == 0 and current_signal == 1: # Buy signal and currently flat for this asset
                        # Determine amount to invest based on position sizing
                        amount_to_invest = 0
                        if position_sizing_method == 'Full Equity':
                            amount_to_invest = self.portfolio.loc[current_index_label, 'cash'] # Use all available cash
                        elif position_sizing_method == 'Fixed Amount':
                            amount_to_invest = position_sizing_value
                        elif position_sizing_method == 'Percentage of Equity':
                            amount_to_invest = self.portfolio.loc[current_index_label, 'total'] * position_sizing_value # Use percentage of total equity

                        # Ensure we don't try to invest more cash than available
                        amount_to_invest = min(amount_to_invest, self.portfolio.loc[current_index_label, 'cash'])

                        # Buy as many shares as possible with the determined amount at the current day's open price
                        buy_price = current_open
                        shares_to_buy = 0
                        if buy_price > 0 and amount_to_invest > 0: # Avoid division by zero or investing zero
                             shares_to_buy = amount_to_invest // buy_price # Integer division for whole shares

                        if shares_to_buy > 0:
                            cost = shares_to_buy * buy_price
                            self.portfolio.loc[current_index_label, 'cash'] -= cost # Deduct cash from portfolio
                            self.asset_holdings[symbol].loc[current_index_label] += shares_to_buy # Add holdings for this asset
                            asset_state['position'] = 1
                            asset_state['entry_price'] = buy_price
                            shares_held = self.asset_holdings[symbol].loc[current_index_label] # Update local shares_held
                            trade_count += 1

                            # Set Stop Loss and Take Profit levels based on entry price
                            if stop_loss_pct > 0:
                                asset_state['stop_loss_price'] = entry_price * (1 - stop_loss_pct)
                            if take_profit_pct > 0:
                                asset_state['take_profit_price'] = entry_price * (1 + take_profit_pct)

                            self.trades.append({
                                'symbol': symbol,
                                'type': 'Buy',
                                'date': current_index_label,
                                'price': buy_price,
                                'shares': shares_to_buy,
                                'cost': cost,
                                'stop_loss': asset_state['stop_loss_price'], # Record SL/TP for this trade
                                'take_profit': asset_state['take_profit_price']
                            })

                    elif position == 1 and current_signal == -1: # Sell signal and currently long for this asset
                        # Sell all holdings at the current day's open price
                        # Using open price for exit assumption
                        sell_price = current_open
                        if shares_held > 0 and sell_price > 0: # Ensure we have shares and price is positive
                            revenue = shares_held * sell_price
                            profit_loss = (sell_price - entry_price) * shares_held
                            self.portfolio.loc[current_index_label, 'cash'] += revenue # Add cash to portfolio
                            self.asset_holdings[symbol].loc[current_index_label] = 0.0 # Zero out holdings for this asset
                            asset_state['position'] = 0
                            asset_state['entry_price'] = 0.0
                            asset_state['stop_loss_price'] = None # Reset SL/TP after exiting
                            asset_state['take_profit_price'] = None
                            self.trades.append({
                                'symbol': symbol,
                                'type': 'Sell',
                                'date': current_index_label,
                                'price': sell_price,
                                'shares': shares_held, # Record shares sold
                                'revenue': revenue,
                                'profit_loss': profit_loss,
                                'exit_type': 'Signal' # Record exit type
                            })
                            shares_held = 0.0 # Update local shares_held after selling


                # --- Update Total Portfolio Value at the end of the day ---
                # Portfolio total = cash + sum of (holdings * close price) for all assets
                total_holdings_value = sum(self.asset_holdings[symbol].loc[current_index_label] * aligned_data[symbol]['Close'].loc[current_index_label]
                                           for symbol in aligned_data.keys() if current_index_label in aligned_data[symbol].index) # Ensure data exists for the day
                self.portfolio.loc[current_index_label, 'total'] = self.portfolio.loc[current_index_label, 'cash'] + total_holdings_value


        except KeyError as e:
            print(f"\n--- KeyError Debug ---")
            print(f"KeyError during backtesting at iteration {i}: {e}")
            print(f"Problematic index value: {current_index_label}")
            print(f"Type of problematic index value: {type(current_index_label)}")
            print(f"Common index head:\n{common_index[:5]}")
            print(f"Common index tail:\n{common_index[-5:]}")
            print(f"Portfolio index head:\n{self.portfolio.index[:5]}")
            print(f"Portfolio index tail:\n{self.portfolio.index[-5:]}")
            print(f"Data keys: {list(aligned_data.keys())}")
            print("----------------------\n")
            return {"error": f"Backtesting stopped due to KeyError: {e}. Check console for details."}
        except Exception as e:
            print(f"\n--- General Error Debug ---")
            print(f"An unexpected error occurred during backtesting at iteration {i}: {e}")
            print(f"Current index label: {current_index_label}")
            print(f"Type of current index label: {type(current_index_label)}")
            # Attempt to print data/portfolio for the current day if indices match
            if current_index_label in common_index:
                 print(f"Current Portfolio Row:\n{self.portfolio.loc[current_index_label] if current_index_label in self.portfolio.index else 'N/A'}")
                 for symbol in aligned_data.keys():
                      if current_index_label in aligned_data[symbol].index:
                           print(f"Current Data Row for {symbol}:\n{aligned_data[symbol].loc[current_index_label]}")
            print("---------------------------\n")
            return {"error": f"Backtesting stopped due to an unexpected error: {e}. Check console for details."}


        # --- Calculate Portfolio-Level Performance Metrics ---
        # Ensure portfolio is not empty before calculating metrics
        if self.portfolio is None or self.portfolio.empty:
             return {"error": "Portfolio history is empty after backtesting."}

        total_return = (self.portfolio['total'].iloc[-1] / self.initial_capital) - 1
        cumulative_returns = self.portfolio['total'] / self.initial_capital
        daily_returns = cumulative_returns.pct_change().dropna()

        # Annualization factor (assuming daily data, 252 trading days)
        annualization_factor = 252

        # Annualized Return
        # Calculate the number of years in the backtest period
        time_span_days = (common_index[-1] - common_index[0]).days if len(common_index) > 1 else 0
        time_span_years = time_span_days / 365.25 if time_span_days > 0 else (1 if len(common_index) > 0 else 0) # Avoid division by zero

        if time_span_years > 0 and self.initial_capital > 0:
             annualized_return = (1 + total_return)**(1 / time_span_years) - 1
        else:
             annualized_return = 0.0


        # Sharpe Ratio
        # Excess daily returns over risk-free rate
        excess_daily_returns = daily_returns - (self._risk_free_rate / annualization_factor)
        if excess_daily_returns.std() != 0:
            sharpe_ratio = excess_daily_returns.mean() / excess_daily_returns.std() * np.sqrt(annualization_factor)
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio
        # Calculate downside deviation (standard deviation of negative returns)
        downside_returns = excess_daily_returns[excess_daily_returns < 0]
        if downside_returns.std() != 0:
            sortino_ratio = excess_daily_returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
        else:
            sortino_ratio = 0.0 # Avoid division by zero

        # Calmar Ratio
        # Calculate the maximum drawdown
        # We use cumulative returns relative to the initial capital
        cumulative_returns_abs = self.portfolio['total']
        # Calculate the running maximum of the cumulative returns
        running_max = cumulative_returns_abs.cummax()
        # Calculate the drawdown
        drawdown = running_max - cumulative_returns_abs
        # Calculate the maximum drawdown value
        max_drawdown_value = drawdown.max()
        # Calculate the maximum drawdown percentage relative to the peak equity
        max_drawdown_pct = max_drawdown_value / running_max.max() if running_max.max() != 0 else 0.0


        # Calmar Ratio
        if max_drawdown_value != 0:
             calmar_ratio = annualized_return / max_drawdown_value # Using absolute drawdown value
             # Some definitions use percentage drawdown, let's use absolute for now
             # calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct != 0 else 0.0
        else:
             calmar_ratio = np.inf if annualized_return > 0 else ( -np.inf if annualized_return < 0 else 0.0) # Infinite if no drawdown and positive return


        results = {
            "initial_capital": self.initial_capital,
            "final_capital": self.portfolio['total'].iloc[-1],
            "total_return": total_return,
            "annualized_return": annualized_return, # Added annualized return
            "cumulative_returns": cumulative_returns,
            "daily_returns": daily_returns,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio, # Added Sortino ratio
            "max_drawdown_pct": max_drawdown_pct, # Renamed for clarity
            "max_drawdown_value": max_drawdown_value, # Added absolute drawdown value
            "calmar_ratio": calmar_ratio, # Added Calmar ratio
            "trade_count": trade_count, # This is total trades across all assets
            "portfolio_history": self.portfolio,
            "trades": self.trades # This list contains trades for all assets
        }

        return results

    def plot_results(self, results: dict):
        """
        Plots the backtest results (portfolio equity curve).
        Note: Plotting price with signals is more complex for multi-asset and is not included here.

        Args:
            results (dict): The results dictionary from run_backtest.
        """
        if "portfolio_history" not in results:
            print("Cannot plot: Portfolio history not available in results.")
            return

        portfolio_history = results["portfolio_history"]

        # Plot equity curve
        equity_fig = plot_equity_curve(portfolio_history['total'], self.initial_capital)
        st.pyplot(equity_fig)
        plt.close(equity_fig) # Close figure to free memory

        # Note: Plotting individual asset performance or price with signals
        # for multi-asset backtests would require more complex plotting logic.

