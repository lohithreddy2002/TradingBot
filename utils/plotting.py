import pandas as pd
import matplotlib.pyplot as plt # Keep import for plot_price_with_signals
import plotly.graph_objects as go # Import Plotly

def plot_equity_curve(equity_curves_dict: dict, initial_capital: float):
    """
    Plots comparative equity curves from a dictionary of pandas Series using Plotly for interactivity.
    This function now handles plotting one or more equity curves from a dictionary.

    Args:
        equity_curves_dict (dict): A dictionary where keys are strategy names
                                   (strings) and values are pandas Series
                                   representing the equity curve over time.
                                   The Series index should be a DatetimeIndex.
        initial_capital (float): The starting capital for the backtest. Used
                                 to plot a baseline.

    Returns:
        plotly.graph_objects.Figure or None: The Plotly figure object if plotting
                                             is successful, otherwise None.
    """
    # Check if the input is a dictionary and is not empty
    if not isinstance(equity_curves_dict, dict) or not equity_curves_dict:
        print(f"Error: Input to plot_equity_curve must be a non-empty dictionary, but received {type(equity_curves_dict)}")
        return None

    fig = go.Figure()

    plotted_curves_count = 0
    all_dates = []

    for strategy_name, equity_curve in equity_curves_dict.items():
        # Check if the value is a pandas Series, is not empty, and has a DatetimeIndex
        if isinstance(equity_curve, pd.Series) and not equity_curve.empty and isinstance(equity_curve.index, pd.DatetimeIndex):
            try:
                # Ensure the index is sorted before plotting
                if not equity_curve.index.is_monotonic_increasing:
                     equity_curve = equity_curve.sort_index()

                # Add trace for the equity curve
                fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name=strategy_name))
                plotted_curves_count += 1
                all_dates.extend(equity_curve.index.tolist())

            except Exception as e:
                print(f"Warning: Could not plot equity curve for '{strategy_name}': {e}")
        else:
            # Provide more specific warnings
            if not isinstance(equity_curve, pd.Series):
                 print(f"Warning: Equity curve for '{strategy_name}' is not a pandas Series (type: {type(equity_curve)}) and will not be plotted.")
            elif equity_curve.empty:
                 print(f"Warning: Equity curve for '{strategy_name}' is empty and will not be plotted.")
            elif not isinstance(equity_curve.index, pd.DatetimeIndex):
                 print(f"Warning: Equity curve for '{strategy_name}' does not have a DatetimeIndex and will not be plotted.")


    # Add the initial capital line if at least one curve was plotted
    if plotted_curves_count > 0:
        if all_dates:
             start_date = min(all_dates)
             end_date = max(all_dates)
             fig.add_trace(go.Scatter(x=[start_date, end_date], y=[initial_capital, initial_capital], mode='lines', name='Initial Capital', line=dict(dash='dash', color='black'))) # Black dashed line
        else:
             # Fallback if somehow no dates were collected (shouldn't happen if plotted_curves_count > 0)
             print("Warning: Could not determine date range for initial capital line.")


        # Update layout for interactivity
        fig.update_layout(
            title='Comparative Equity Curves',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified' # Show tooltip for all traces at the same x-value
        )
        return fig
    else:
        print("No valid equity curves were provided for plotting.")
        return None # Return None if no curves were plotted

def plot_price_with_signals(price_series: pd.Series, signals: pd.Series):
    """
    Plots the price series with buy and sell signals using Matplotlib.

    Args:
        price_series (pd.Series): A pandas Series of the asset's closing price.
        signals (pd.Series): A pandas Series of trading signals (1 for buy, -1 for sell).

    Returns:
        matplotlib.figure.Figure or None: The matplotlib figure object if plotting
                                          is successful, otherwise None.
    """
    # Basic checks for input types and non-empty Series
    if not isinstance(price_series, pd.Series) or price_series.empty:
        print("Error: Input price_series must be a non-empty pandas Series.")
        return None
    if not isinstance(signals, pd.Series) or signals.empty:
        print("Error: Input signals must be a non-empty pandas Series.")
        return None
    if not price_series.index.equals(signals.index):
         print("Error: Price series and signals must have the same index.")
         return None


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_series.index, price_series, label='Close Price', alpha=0.7)

    # Plot buy signals
    buy_signals = price_series[signals == 1]
    if not buy_signals.empty:
        ax.plot(buy_signals.index, buy_signals, '^', markersize=10, color='g', lw=0, label='Buy Signal')

    # Plot sell signals
    sell_signals = price_series[signals == -1]
    if not sell_signals.empty:
        ax.plot(sell_signals.index, sell_signals, 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    ax.set_title('Price with Trading Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig # Return figure for Streamlit
