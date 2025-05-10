import streamlit as st
import pandas as pd
import datetime
import sys
import os
import matplotlib.pyplot as plt # Keep import if plot_price_with_signals is used
from streamlit_searchbox import st_searchbox
import plotly.graph_objects as go # Import Plotly for interactive charts
import json # Import json for saving metrics and trades


# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules from the project structure
from data_sources.base_data_source import BaseDataSource
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource # Import specific data source for searchbox
from strategies.base_strategy import BaseStrategy
from backtester.backtester import Backtester
from utils.plotting import plot_equity_curve, plot_price_with_signals


# --- Helper function for symbol searchbox calling the data source method ---
def search_symbols_single(search_term: str) -> list[str]:
    """
    Calls the search_symbols method of the currently selected data source for the single asset page.
    Accesses AVAILABLE_DATA_SOURCES from Streamlit's session state.
    """
    # Ensure AVAILABLE_DATA_SOURCES is in session state
    AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
    if not AVAILABLE_DATA_SOURCES:
        # This should ideally not happen if app.py runs first, but as a safeguard
        print("Error: AVAILABLE_DATA_SOURCES not found in session state.")
        return ["Error: Data sources not loaded."]

    selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])
    DataSourceClass = AVAILABLE_DATA_SOURCES.get(selected_data_source_name)

    if DataSourceClass and hasattr(DataSourceClass, 'search_symbols') and callable(getattr(DataSourceClass, 'search_symbols')):
        data_source_instance = DataSourceClass()
        return data_source_instance.search_symbols(search_term)
    else:
        # Fallback for data sources that don't implement search_symbols
        print(f"Warning: Selected data source '{selected_data_source_name}' does not have a 'search_symbols' method.")
        # If the data source has get_available_symbols, filter that list
        if DataSourceClass and hasattr(DataSourceClass, 'get_available_symbols') and callable(getattr(DataSourceClass, 'get_available_symbols')):
             data_source_instance = DataSourceClass()
             all_symbols = data_source_instance.get_available_symbols()
             if not search_term:
                  return all_symbols[:20]
             search_term_lower = search_term.lower()
             return [symbol for symbol in all_symbols if search_term_lower in symbol.lower()][:20]
        else:
             return [f"Search not available for {selected_data_source_name}"]


# --- UI Elements and Logic for the Single Strategy Backtest Page ---
# This code will be executed directly by Streamlit when the page is selected.

st.title("Single Strategy Backtest")

# Retrieve available data sources and strategies from session state
AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})

if not AVAILABLE_DATA_SOURCES or not AVAILABLE_STRATEGIES:
    st.error("Available data sources or strategies not found in session state. Please run the app from app.py.")
    st.stop()


st.header("Backtest Parameters")

# Use a container for Data Source, Date Range, and Initial Capital
with st.container(border=True):
    st.subheader("Data Selection and Capital")
    selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])
    DataSourceClass = AVAILABLE_DATA_SOURCES.get(selected_data_source_name)
    if not DataSourceClass:
         st.error(f"Data source '{selected_data_source_name}' not found.")
         st.stop()
    data_source = DataSourceClass() # Instantiate the selected data source
    st.write(f"Using Data Source: **{selected_data_source_name}** (Change in Settings)")

    col_date_start, col_date_end = st.columns(2)
    with col_date_start:
        today = datetime.date.today()
        start_date = st.date_input("Start Date", today - datetime.timedelta(days=365*2), key='single_start_date')
    with col_date_end:
        end_date = st.date_input("End Date", today, key='single_end_date')

    # Initial Capital - Get default from settings, allow override
    default_initial_capital = st.session_state.get('initial_capital', 100000.0)
    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=default_initial_capital, step=1000.0, key='single_initial_capital_override')


    st.subheader("Symbol Selection")
    selected_symbol = st_searchbox(
        search_symbols_single,
        label="Search and Select Symbol",
        key="single_symbol_searchbox",
    )
    st.write(f"Selected Symbol: **{selected_symbol}**")


# Use a container for Strategy and Parameters
with st.container(border=True):
    st.subheader("Strategy Configuration")
    selected_strategy_name = st.selectbox("Select Strategy", list(AVAILABLE_STRATEGIES.keys()), key='single_strategy') # This sets st.session_state['single_strategy']
    StrategyClass = AVAILABLE_STRATEGIES.get(selected_strategy_name) # Use .get for safety


    # Use an expander for Strategy Parameters
    with st.expander("Configure Strategy Parameters"):
        strategy_params = {}
        if StrategyClass: # Only attempt to load params if StrategyClass is found
            try:
                temp_strategy = StrategyClass()
                if hasattr(temp_strategy, 'get_params') and callable(getattr(temp_strategy, 'get_params')):
                    default_params = temp_strategy.get_params()
                    if default_params:
                        param_cols = st.columns(min(len(default_params), 3))
                        param_col_index = 0
                        for param_name, default_value in default_params.items():
                            current_col = param_cols[param_col_index % len(param_cols)]
                            with current_col:
                                # Get the current value from session state or use default
                                # Need to handle potential missing keys in session state for new parameters or first run
                                # Use a unique key for each parameter input based on strategy name and parameter name
                                unique_key = f'single_param_{selected_strategy_name}_{param_name}'
                                current_value = st.session_state.get(unique_key, default_value)


                                if isinstance(default_value, int):
                                    strategy_params[param_name] = st.number_input(param_name, min_value=1, value=int(current_value), step=1, key=f'{unique_key}_int')
                                elif isinstance(default_value, float):
                                     strategy_params[param_name] = st.number_input(param_name, value=float(current_value), step=0.1, format="%.2f", key=f'{unique_key}_float')
                                elif isinstance(default_value, bool):
                                     strategy_params[param_name] = st.checkbox(param_name, value=bool(current_value), key=f'{unique_key}_bool')
                                else:
                                     strategy_params[param_name] = st.text_input(param_name, str(current_value), key=f'{unique_key}_text')

                                # Store parameter value in session state for persistence
                                st.session_state[unique_key] = strategy_params[param_name]
                            param_col_index += 1 # Move to the next column


                    else:
                        st.info("This strategy has no configurable parameters.")
                        strategy_params = {} # Ensure strategy_params is an empty dict if no params

            except Exception as e:
                st.warning(f"Could not load parameters for {selected_strategy_name}: {e}")
                strategy_params = {} # Ensure strategy_params is an empty dict if parameter loading fails
        else:
             st.warning(f"Strategy class not found for {selected_strategy_name}.")
             strategy_params = {"Error": "Strategy class not found"} # Indicate error

    # Attempt to instantiate the strategy after parameters are defined
    strategy_instance = None
    if StrategyClass and "Error" not in strategy_params:
         try:
             strategy_instance = StrategyClass(**strategy_params)
         except Exception as e:
              st.error(f"Error initializing strategy {selected_strategy_name}: {e}")


# Use a container for Risk Management and Position Sizing
with st.container(border=True):
    st.subheader("Risk and Position Management")
    col_sl, col_tp = st.columns(2)
    with col_sl:
        stop_loss_pct = st.number_input("Global Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.2f", key='single_stop_loss_pct') / 100.0
    with col_tp:
        take_profit_pct = st.number_input("Global Take Profit (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key='single_take_profit_pct') / 100.0

    st.subheader("Position Sizing")
    position_sizing_method = st.selectbox(
        "Method",
        ['Full Equity', 'Fixed Amount', 'Percentage of Equity'],
        key='single_position_sizing_method'
    )
    position_sizing_value = 0.0
    if position_sizing_method == 'Fixed Amount':
        position_sizing_value = st.number_input("Amount ($)", min_value=1.0, value=1000.0, step=100.0, key='single_position_sizing_amount')
    elif position_sizing_method == 'Percentage of Equity':
        position_sizing_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f", key='single_position_sizing_percentage') / 100.0


# --- Run Backtest Button ---
# Disable the run button if no symbol or strategy is selected, or if initial capital is invalid,
# or if position sizing is invalid, or if the strategy instance could not be created.
is_disabled = not selected_symbol or strategy_instance is None or (position_sizing_method != 'Full Equity' and position_sizing_value <= 0)

st.markdown("---") # Add a separator before the button

# Add a placeholder for the Save Results button
save_button_placeholder = st.empty()

# --- Debugging: Show state before Run button check ---
# print(f"DEBUG: State before Run button check - last_single_backtest_results: {'present' if 'last_single_backtest_results' in st.session_state and st.session_state.get('last_single_backtest_results') is not None else 'not present/None'}, last_single_backtest_config: {'present' if 'last_single_backtest_config' in st.session_state and st.session_state.get('last_single_backtest_config') is not None else 'not present/None'}")


if st.button("Run Backtest", key='run_single_backtest_main', disabled=is_disabled):
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
        st.toast("Error: Start date must be before end date.", icon="❌")
    elif not selected_symbol:
         st.warning("Please select a symbol to run the backtest.")
         st.toast("Please select a symbol to run the backtest.", icon="⚠️")
    elif position_sizing_method != 'Full Equity' and position_sizing_value <= 0:
         st.warning(f"Please enter a valid {position_sizing_method} value.")
         st.toast(f"Please enter a valid {position_sizing_method} value.", icon="⚠️")
    elif strategy_instance is None: # Added check here as well
         st.warning("Strategy could not be initialized. Check parameters.")
         st.toast("Strategy could not be initialized. Check parameters.", icon="❌")
    else:
        st.subheader("Backtest Results")

        # --- Fetch Data for the single selected symbol ---
        # st.info(f"Fetching data for {selected_symbol} from {start_date} to {end_date} using {selected_data_source_name}...")
        st.toast(f"Fetching data for {selected_symbol}...", icon="⏳")
        try:
            data_df = data_source.fetch_data(selected_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if data_df.empty:
                st.warning("No data fetched for the selected criteria.")
                st.toast("No data fetched for the selected criteria.", icon="⚠️")
                data = {}
                # Clear previous results if data fetching fails
                if 'last_single_backtest_results' in st.session_state:
                    del st.session_state['last_single_backtest_results']
                if 'last_single_backtest_config' in st.session_state: # Also clear config on data fetch error
                     del st.session_state['last_single_backtest_config']
            else:
                # st.success("Data fetched successfully.")
                st.toast("Data fetched successfully!", icon="✅")
                st.dataframe(data_df.head(), key='single_data_head')
                data = {selected_symbol: data_df}

        except Exception as e:
            st.error(f"An error occurred during data fetching: {e}")
            st.toast(f"Error fetching data: {e}", icon="❌")
            data = {}
            # Clear previous results if data fetching fails
            if 'last_single_backtest_results' in st.session_state:
                del st.session_state['last_single_backtest_results']
            if 'last_single_backtest_config' in st.session_state: # Also clear config on data fetch error
                 del st.session_state['last_single_backtest_config']


        if not data:
             st.error("No data available for backtesting after fetching.")
             st.toast("No data available for backtesting.", icon="❌")
             st.stop()


        # --- Run Backtest ---
        # st.info(f"Running backtest with {strategy_instance.name} on {list(data.keys())}...")
        st.toast(f"Running backtest with {strategy_instance.name}...")
        backtester = Backtester(initial_capital=initial_capital)
        try:
            results = backtester.run_backtest(
                data.copy(),
                strategy_instance,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_sizing_method=position_sizing_method,
                position_sizing_value=position_sizing_value
            )

            if "error" in results:
                st.error(f"Backtest Error: {results['error']}")
                st.toast(f"Backtest failed: {results['error']}", icon="❌")
                # Clear previous results if backtest fails
                if 'last_single_backtest_results' in st.session_state:
                    del st.session_state['last_single_backtest_results']
                if 'last_single_backtest_config' in st.session_state: # Also clear config on backtest error
                     del st.session_state['last_single_backtest_config']

            else:
                # st.success("Backtest completed.")
                st.toast("Backtest completed successfully!")

                # Store results in session state for saving
                st.session_state['last_single_backtest_results'] = results
                # Store current configuration used for this backtest run for saving purposes
                st.session_state['last_single_backtest_config'] = {
                    "strategy_name": selected_strategy_name,
                    "strategy_parameters": strategy_params, # Use the actual params used
                    "data_source": selected_data_source_name,
                    "symbol": selected_symbol,
                    "start_date": start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime.date) else str(start_date),
                    "end_date": end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else str(end_date),
                    "initial_capital": initial_capital,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "position_sizing_method": position_sizing_method,
                    "position_sizing_value": position_sizing_value
                }

                # --- Debugging: Show state after successful backtest ---
                # print(f"DEBUG: State after successful backtest - last_single_backtest_results: {'present' if 'last_single_backtest_results' in st.session_state and st.session_state.get('last_single_backtest_results') is not None else 'not present/None'}, last_single_backtest_config: {'present' if 'last_single_backtest_config' in st.session_state and st.session_state.get('last_single_backtest_config') is not None else 'not present/None'}")


                # --- Display Metrics ---
                st.subheader("Performance Metrics")
                # Safely get metrics with default values
                initial_cap = results.get('initial_capital', 0)
                final_cap = results.get('final_capital', 0)
                total_ret = results.get('total_return', 0)
                annual_ret = results.get('annualized_return', 0)
                sharpe = results.get('sharpe_ratio', 0)
                sortino = results.get('sortino_ratio', 0)
                calmar = results.get('calmar_ratio', 0)
                max_dd_pct = results.get('max_drawdown_pct', 0)
                max_dd_value = results.get('max_drawdown_value', 0)
                trade_count = results.get('trade_count', 0)


                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Initial Capital", f"${initial_cap:,.2f}")
                col2.metric("Final Capital", f"${final_cap:,.2f}")
                col3.metric("Total Return", f"{total_ret:.2%}")
                col4.metric("Annualized Return", f"{annual_ret:.2%}")

                col5, col6, col7 = st.columns(3)
                col5.metric("Sharpe Ratio", f"{sharpe:.2f}")
                col6.metric("Sortino Ratio", f"{sortino:.2f}")
                col7.metric("Calmar Ratio", f"{calmar:.2f}")

                col8, col9 = st.columns(2)
                col8.metric("Maximum Drawdown (%)", f"{max_dd_pct:.2%}")
                col9.metric("Number of Trades", f"{trade_count:,}")


                # --- Plot Equity Curve (Interactive) ---
                st.subheader("Equity Curve")
                # The plot_equity_curve function now expects a dictionary
                # For single backtest, wrap the equity curve in a dictionary
                equity_curves_for_plotting = {}
                if 'portfolio_history' in results and results['portfolio_history'] is not None and 'total' in results['portfolio_history']:
                     portfolio_total = results['portfolio_history']['total']
                     if isinstance(portfolio_total, pd.Series) and not portfolio_total.empty:
                          # Ensure index is DatetimeIndex and sorted before passing
                          if not isinstance(portfolio_total.index, pd.DatetimeIndex):
                               try:
                                    portfolio_total.index = pd.to_datetime(portfolio_total.index)
                               except Exception as e:
                                    st.warning(f"Could not convert index to DatetimeIndex for equity curve: {e}")
                                    portfolio_total = None # Discard if conversion fails

                          if portfolio_total is not None and not portfolio_total.index.is_monotonic_increasing:
                               portfolio_total = portfolio_total.sort_index()

                          if portfolio_total is not None:
                               # Use the strategy name as the key for the dictionary
                               equity_curves_for_plotting[selected_strategy_name] = portfolio_total
                          else:
                               st.warning("No valid equity curve data available for plotting.")
                     else:
                          st.warning("No valid equity curve data available for plotting.")
                else:
                     st.info("No portfolio history found in backtest results for plotting.")


                if equity_curves_for_plotting:
                     try:
                         # Use the updated plot_equity_curve function which is interactive
                         fig = plot_equity_curve(equity_curves_for_plotting, initial_capital=initial_capital)
                         if fig: # Check if the plotting function returned a figure
                             st.plotly_chart(fig) # Use st.plotly_chart for interactive plots
                             # No need to close Plotly figures with plt.close()
                         else:
                             st.warning("Could not generate interactive equity curve plot.")
                     except Exception as e:
                         st.error(f"An error occurred while plotting equity curve: {e}")


                # --- Plot Price with Signals (if implemented in strategy and backtester) ---
                # This part assumes your backtester results might include signals or trades
                # and your strategy might have a method to get signals.
                # You might need to adjust this based on your actual Backtester and Strategy implementations.
                # Example (requires signals to be stored in results or calculable from trades):
                # if 'signals' in results and results['signals'] is not None:
                #     st.subheader("Price with Signals")
                #     try:
                #         # Assuming plot_price_with_signals exists and works with your signal format
                #         price_series_for_plotting = data[selected_symbol]['Close'] # Assuming 'Close' is available for the selected symbol
                #         signals_for_plotting = results['signals'] # Assuming signals are in results
                #         fig_signals = plot_price_with_signals(price_series_for_plotting, signals_for_plotting)
                #         if fig_signals:
                #             st.pyplot(fig_signals) # Assuming plot_price_with_signals is Matplotlib-based
                #             plt.close(fig_signals)
                #         else:
                #             st.warning("Could not generate price with signals plot.")
                #     except Exception as e:
                #         st.error(f"An error occurred while plotting price with signals: {e}")
                #         plt.close('all') # Close any potentially created figure


                # --- Display Trade Log ---
                if 'trades' in results and results['trades']:
                    with st.expander("View Trade Log"):
                        # Ensure trades data is a list of dictionaries before creating DataFrame
                        if isinstance(results['trades'], list) and all(isinstance(trade, dict) for trade in results['trades']):
                            trades_df = pd.DataFrame(results['trades'])
                            # Format timestamp columns if they exist
                            for col in ['entry_time', 'exit_time']: # Adjust column names as per your trades structure
                                if col in trades_df.columns:
                                    try:
                                        # Attempt to convert to datetime, handle errors
                                        trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                                    except Exception as e:
                                        st.warning(f"Could not format trade log timestamp column '{col}': {e}")
                                        # Keep original data if formatting fails

                            st.dataframe(trades_df, key='single_trade_log')
                        else:
                            st.warning("Trade log data is not in the expected format.")
                            st.json(results['trades']) # Display raw data for debugging
                else:
                     st.info("No trades recorded for this backtest.")

        except Exception as e:
            st.error(f"An unexpected error occurred during backtest execution or results processing: {e}")
            st.toast(f"An unexpected error occurred: {e}", icon="❌")
            # Clear previous results if backtest fails
            if 'last_single_backtest_results' in st.session_state:
                del st.session_state['last_single_backtest_results']
            if 'last_single_backtest_config' in st.session_state:
                 del st.session_state['last_single_backtest_config']


  

# --- Save Results Button ---
# Display the save button only if there are backtest results AND config in session state and they are not None
if 'last_single_backtest_results' in st.session_state and st.session_state.get('last_single_backtest_results') is not None \
   and 'last_single_backtest_config' in st.session_state and st.session_state.get('last_single_backtest_config') is not None:

    # --- Debugging: Show state inside save button block ---
    # print("DEBUG: Inside save button block - session state variables are present and not None.")

    results_to_save = st.session_state['last_single_backtest_results']
    config_used = st.session_state['last_single_backtest_config']

    # Retrieve selected symbol and strategy name from the config stored in session state
    # This is safer than relying on the local variables which might not be defined
    # in all rerun scenarios when this block is executed.
    symbol_for_filename = config_used.get('symbol', 'UnknownSymbol')
    strategy_name_for_filename = config_used.get('strategy_name', 'UnknownStrategy')


    # Prepare data for download
    metrics_data = {
        "Initial Capital": results_to_save.get('initial_capital', 0),
        "Final Capital": results_to_save.get('final_capital', 0),
        "Total Return": results_to_save.get('total_return', 0),
        "Annualized Return": results_to_save.get('annualized_return', 0),
        "Sharpe Ratio": results_to_save.get('sharpe_ratio', 0),
        "Sortino Ratio": results_to_save.get('sortino_ratio', 0),
        "Calmar Ratio": results_to_save.get('calmar_ratio', 0),
        "Max Drawdown (%)": results_to_save.get('max_drawdown_pct', 0),
        "Max Drawdown ($)": results_to_save.get('max_drawdown_value', 0),
        "Trade Count": results_to_save.get('trade_count', 0),
        # Include configuration details for completeness in the metrics JSON
        "Configuration": {
            "Strategy": config_used.get('strategy_name'),
            "Strategy Parameters": config_used.get('strategy_parameters'),
            "Data Source": config_used.get('data_source'),
            "Symbol": config_used.get('symbol'),
            "Start Date": config_used.get('start_date'),
            "End Date": config_used.get('end_date'),
            "Initial Capital": config_used.get('initial_capital'),
            "Stop Loss (%)": config_used.get('stop_loss_pct') * 100 if config_used.get('stop_loss_pct') is not None else 'N/A',
            "Take Profit (%)": config_used.get('take_profit_pct') * 100 if config_used.get('take_profit_pct') is not None else 'N/A',
            "Position Sizing Method": config_used.get('position_sizing_method'),
            "Position Sizing Value": config_used.get('position_sizing_value') if config_used.get('position_sizing_method') != 'Full Equity' else 'N/A'
        }
    }
    metrics_json = json.dumps(metrics_data, indent=4)

    equity_curve_df = pd.DataFrame()
    if 'portfolio_history' in results_to_save and results_to_save['portfolio_history'] is not None and 'total' in results_to_save['portfolio_history']:
        portfolio_total = results_to_save['portfolio_history']['total']
        if isinstance(portfolio_total, pd.Series) and not portfolio_total.empty:
             # Ensure index is DatetimeIndex and sorted before creating DataFrame for save
             if not isinstance(portfolio_total.index, pd.DatetimeIndex):
                  try:
                       portfolio_total.index = pd.to_datetime(portfolio_total.index)
                  except Exception as e:
                       st.warning(f"Could not convert index to DatetimeIndex for equity curve for saving: {e}")
                       portfolio_total = None # Discard if conversion fails

             if portfolio_total is not None and not portfolio_total.index.is_monotonic_increasing:
                  portfolio_total = portfolio_total.sort_index()

             if portfolio_total is not None:
                  equity_curve_df = portfolio_total.reset_index()
                  equity_curve_df.columns = ['Date', 'Equity']
                  # Ensure Date is in a standard format for CSV
                  equity_curve_df['Date'] = equity_curve_df['Date'].dt.strftime('%Y-%m-%d')
             else:
                  st.warning("No valid equity curve data available for saving.")


    trades_df = pd.DataFrame()
    if 'trades' in results_to_save and results_to_save['trades'] is not None:
        # Assuming trades is a list of dictionaries or similar structure
        # You might need to adjust this based on the exact format of your trades data
        if isinstance(results_to_save['trades'], list) and all(isinstance(trade, dict) for trade in results_to_save['trades']):
            try:
                trades_df = pd.DataFrame(results_to_save['trades'])
                # Convert timestamp columns to datetime if they exist
                for col in ['entry_time', 'exit_time']: # Adjust column names as per your trades structure
                     if col in trades_df.columns:
                          # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
                          trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                st.warning(f"Could not convert trades data to DataFrame for saving: {e}")
                trades_df = pd.DataFrame([{"Error": f"Could not process trades data: {e}"}]) # Indicate error
        else:
             st.warning("Trades data is not in the expected format for saving.")
             trades_df = pd.DataFrame([{"Error": "Trades data format is incorrect"}])


    # Use the placeholder to render the download buttons
    with save_button_placeholder.container():
        st.markdown("---") # Add a separator before save buttons
        st.subheader("Download Results")
        col_metrics, col_equity, col_trades = st.columns(3)

        with col_metrics:
            st.download_button(
                label="Metrics (JSON)",
                data=metrics_json,
                # Use values from session state for file naming
                file_name=f"{symbol_for_filename}_{strategy_name_for_filename}_metrics.json",
                mime="application/json",
                key='download_single_metrics_json',
                on_click=lambda: st.toast("Metrics JSON prepared for download!", icon="💾")
            )

        with col_equity:
            if not equity_curve_df.empty:
                 st.download_button(
                    label="Equity Curve (CSV)",
                    data=equity_curve_df.to_csv(index=False).encode('utf-8'),
                     # Use values from session state for file naming
                    file_name=f"{symbol_for_filename}_{strategy_name_for_filename}_equity_curve.csv",
                    mime="text/csv",
                    key='download_single_equity_curve_csv',
                    on_click=lambda: st.toast("Equity Curve CSV prepared for download!", icon="💾")
                )
            else:
                 st.info("No equity curve data to download.")


        with col_trades:
            # Check if trades_df was successfully created and does not contain the error column
            if not trades_df.empty and "Error" not in trades_df.columns:
                st.download_button(
                    label="Trades (CSV)",
                    data=trades_df.to_csv(index=False).encode('utf-8'),
                     # Use values from session state for file naming
                    file_name=f"{symbol_for_filename}_{strategy_name_for_filename}_trades.csv",
                    mime="text/csv",
                    key='download_single_trades_csv',
                    on_click=lambda: st.toast("Trades CSV prepared for download!", icon="💾")
                )
            elif "Error" in trades_df.columns:
                 st.info(f"Trades data could not be processed for download: {trades_df.loc[0, 'Error']}")
            else:
                 st.info("No trades data to download.")


# Display warnings below the button if disabled (moved outside the button click block)
# These warnings will persist as long as the conditions are met
# (The warnings inside the button click block are for transient issues during execution)
elif not selected_symbol:
     st.warning("Please select a symbol to run the backtest.")
elif strategy_instance is None:
     st.warning("Strategy could not be initialized. Check parameters.")
elif position_sizing_method != 'Full Equity' and position_sizing_value <= 0:
     st.warning(f"Please enter a valid {position_sizing_method} value.")
