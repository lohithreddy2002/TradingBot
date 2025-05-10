import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Keep import if plot_price_with_signals is used elsewhere
import sys
import os
import datetime
import plotly.graph_objects as go # Import Plotly for interactive charts
import json # Import json for saving metrics and trades


# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules (strategies, data sources, backtester, plotting utility)
from strategies.base_strategy import BaseStrategy
from backtester.backtester import Backtester
from data_sources.base_data_source import BaseDataSource
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource
from streamlit_searchbox import st_searchbox
from utils.plotting import plot_equity_curve, plot_price_with_signals # Import the modified plot_equity_curve and plot_price_with_signals


# --- Helper function for symbol searchbox calling the data source method ---
def search_symbols_compare(search_term: str) -> list[str]:
    """
    Calls the search_symbols method of the currently selected data source for the compare page.
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


# --- UI Elements and Logic for the Compare Strategies Page ---
# This code will be executed directly by Streamlit when the page is selected.

st.title("Compare Strategies")

# Retrieve available strategies and data sources from session state
AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})
AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])


if not AVAILABLE_STRATEGIES:
    st.warning("No trading strategies available for comparison.")
    st.stop()

st.header("Configuration")

# Use a container for Data Source, Date Range, and Symbol Selection
with st.container(border=True):
    st.subheader("Data Selection")
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
        start_date = st.date_input("Backtest Start Date", today - datetime.timedelta(days=365*5), key='compare_start_date')
    with col_date_end:
        end_date = st.date_input("Backtest End Date", today, key='compare_end_date')

    st.subheader("Symbol Selection")
    st.write("Select symbols for multi-asset backtesting:")
    num_symbols = st.number_input("Number of Symbols", min_value=1, value=1, step=1, key='compare_num_symbols')
    selected_symbols = []
    for i in range(num_symbols):
        symbol = st_searchbox(
            search_symbols_compare, # Use the compare search function
            label=f"Search and Select Symbol {i+1}",
            key=f"compare_symbol_searchbox_{i}"
        )
        if symbol:
            selected_symbols.append(symbol)

    st.write(f"Selected Symbols: **{', '.join(selected_symbols) if selected_symbols else 'None'}**")


# Use a container for Strategy Selection and Parameter Configuration
with st.container(border=True):
    st.subheader("Strategy Selection and Parameter Configuration")

    # Allow selecting multiple strategies
    selected_strategy_names = st.multiselect(
        "Select Strategies to Compare",
        list(AVAILABLE_STRATEGIES.keys()),
        key='compare_strategies_select'
    )

    # Store parameter configuration for each strategy in session state
    if 'compare_parameter_config' not in st.session_state:
        st.session_state['compare_parameter_config'] = {}

    # Display and allow configuration of parameters for each selected strategy via expander
    configured_strategy_params = {} # Dictionary to hold configured parameters for backtesting

    if selected_strategy_names:
        st.subheader("Configure Parameters")

        for strategy_name in selected_strategy_names:
            StrategyClass = AVAILABLE_STRATEGIES.get(strategy_name)
            if StrategyClass:
                # Use st.expander for each strategy
                with st.expander(f"Configure {strategy_name}"):
                    if hasattr(StrategyClass, 'get_params') and callable(getattr(StrategyClass, 'get_params')):
                        try:
                            temp_strategy = StrategyClass()
                            default_params = temp_strategy.get_params()
                            if default_params:
                                # Use a unique key for each parameter input based on strategy name and parameter name
                                strategy_params = {}
                                for param_name, default_value in default_params.items():
                                    unique_key = f'compare_param_{strategy_name}_{param_name}'
                                    # Get the current value from session state, defaulting to the strategy's default if not found
                                    current_value = st.session_state['compare_parameter_config'].get(strategy_name, {}).get(param_name, default_value)

                                    if isinstance(default_value, int):
                                        param_value = st.number_input(f"{param_name}", value=int(current_value), step=1, key=unique_key)
                                    elif isinstance(default_value, float):
                                        param_value = st.number_input(f"{param_name}", value=float(current_value), step=0.1, format="%.2f", key=unique_key)
                                    elif isinstance(default_value, bool):
                                         param_value = st.checkbox(f"{param_name}", value=bool(current_value), key=unique_key)
                                    else:
                                        param_value = st.text_input(f"{param_name}", value=str(current_value), key=unique_key)

                                    strategy_params[param_name] = param_value
                                    # Store the configured value in session state
                                    if strategy_name not in st.session_state['compare_parameter_config']:
                                         st.session_state['compare_parameter_config'][strategy_name] = {}
                                    st.session_state['compare_parameter_config'][strategy_name][param_name] = param_value

                                configured_strategy_params[strategy_name] = strategy_params

                            else:
                                st.write("No configurable parameters.")
                                configured_strategy_params[strategy_name] = {} # Store empty dict if no params

                        except Exception as e:
                            st.warning(f"Could not load parameters for {strategy_name}: {e}")
                            configured_strategy_params[strategy_name] = {"Error": str(e)} # Indicate error in config

                    else:
                         st.write("No configurable parameters.")
                         configured_strategy_params[strategy_name] = {} # Store empty dict if no params

            else:
                 st.warning(f"Strategy class not found for {strategy_name}.")
                 configured_strategy_params[strategy_name] = {"Error": "Strategy class not found"}


    # Ensure configured_strategy_params is populated correctly for all selected strategies
    # This handles cases where strategies are selected but the expander hasn't been opened yet
    if selected_strategy_names:
         for name in selected_strategy_names:
              if name not in configured_strategy_params: # If not already populated by the expander interaction
                   if name in st.session_state['compare_parameter_config']:
                        configured_strategy_params[name] = st.session_state['compare_parameter_config'][name]
                   else:
                        # If a strategy was selected but not yet configured and not in session state, use its default params
                        try:
                             default_params_for_unconfigured = AVAILABLE_STRATEGIES.get(name)().get_params()
                             configured_strategy_params[name] = default_params_for_unconfigured
                             # Also store defaults in session state for consistency
                             st.session_state['compare_parameter_config'][name] = default_params_for_unconfigured
                        except Exception as e:
                             st.warning(f"Could not load default parameters for unconfigured strategy {name}: {e}")
                             configured_strategy_params[name] = {"Error": f"Could not load default parameters: {e}"}


# Use a container for Initial Capital, Risk Management, and Position Sizing
with st.container(border=True):
    st.subheader("Capital, Risk, and Position Management")
    # Get default initial capital from settings, allow override
    default_initial_capital = st.session_state.get('initial_capital', 100000.0)
    initial_capital = st.number_input("Initial Capital for Backtests", min_value=1000.0, value=default_initial_capital, step=1000.0, key='compare_initial_capital_override')


    col_sl, col_tp = st.columns(2)
    with col_sl:
        compare_stop_loss_pct = st.number_input("Global Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.2f", key='compare_stop_loss_pct') / 100.0
    with col_tp:
        compare_take_profit_pct = st.number_input("Global Take Profit (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key='compare_take_profit_pct') / 100.0


    st.subheader("Position Sizing")
    compare_position_sizing_method = st.selectbox(
        "Method",
        ['Full Equity', 'Fixed Amount', 'Percentage of Equity'],
        key='compare_position_sizing_method'
    )
    compare_position_sizing_value = 0.0
    if compare_position_sizing_method == 'Fixed Amount':
        compare_position_sizing_value = st.number_input("Amount ($)", min_value=1.0, value=1000.0, step=100.0, key='compare_position_sizing_amount')
    elif compare_position_sizing_method == 'Percentage of Equity':
        compare_position_sizing_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f", key='compare_position_sizing_percentage') / 100.0


# --- Run Backtests Button ---
# Disable the run button if no symbols or strategies are selected, or if initial capital is invalid,
# or if position sizing is invalid, or if any selected strategy has a parameter configuration error.
is_disabled = not selected_symbols or not selected_strategy_names or initial_capital <= 0
if compare_position_sizing_method != 'Full Equity' and compare_position_sizing_value <= 0:
     is_disabled = True
     st.warning(f"Please enter a valid {compare_position_sizing_method} value.")

# Check for parameter configuration errors in any selected strategy
if any("Error" in params for params in configured_strategy_params.values()):
     is_disabled = True
     st.warning("Cannot run backtests due to parameter configuration errors in one or more selected strategies.")
elif not selected_strategy_names: # Also disable if no strategies are selected
     is_disabled = True


st.markdown("---") # Add a separator before the button

# Add a placeholder for the Save Results button
save_button_placeholder = st.empty()


if st.button("Run Comparative Backtests", key='run_compare_backtests', disabled=is_disabled):
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
        st.toast("Error: Start date must be before end date.", icon="❌")
    elif not selected_symbols:
         st.warning("Please select at least one symbol for backtesting.")
         st.toast("Please select at least one symbol for backtesting.", icon="⚠️")
    elif not selected_strategy_names:
         st.warning("Please select at least one strategy to compare.")
         st.toast("Please select at least one strategy to compare.", icon="⚠️")
    elif initial_capital <= 0:
         st.warning("Initial capital must be greater than 0.")
         st.toast("Initial capital must be greater than 0.", icon="⚠️")
    elif compare_position_sizing_method != 'Full Equity' and compare_position_sizing_value <= 0:
         st.warning(f"Please enter a valid {compare_position_sizing_method} value.")
         st.toast(f"Please enter a valid {compare_position_sizing_method} value.", icon="⚠️")
    elif any("Error" in params for params in configured_strategy_params.values()):
         st.warning("Cannot run backtests due to parameter configuration errors.")
         st.toast("Cannot run backtests due to parameter configuration errors.", icon="❌")

    else:
        st.subheader("Backtest Results")

        # --- Fetch Data ---
        st.info(f"Fetching data for {', '.join(selected_symbols)} from {start_date} to {end_date} using {selected_data_source_name}...")
        st.toast(f"Fetching data for {', '.join(selected_symbols)}...", icon="⏳")
        data = {}
        fetch_errors = []
        for symbol in selected_symbols:
            try:
                asset_data = data_source.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if not asset_data.empty:
                    data[symbol] = asset_data
                else:
                    fetch_errors.append(f"No data fetched for {symbol}")
            except Exception as e:
                fetch_errors.append(f"Error fetching data for {symbol}: {e}")

        if fetch_errors:
            for error in fetch_errors:
                st.error(error)
                st.toast(f"Error fetching data: {error}", icon="❌") # Use the error message in the toast
            if not data:
                 st.error("No data available for backtesting after fetching.")
                 st.toast("No data available for backtesting.", icon="❌")
                 st.stop()

        if not data:
             st.error("No data available for backtesting.")
             st.toast("No data available for backtesting.", icon="❌")
             st.stop()


        st.success(f"Data fetched successfully for {list(data.keys())}.")
        st.toast("Data fetched successfully!", icon="✅")
        # Optional: Display head of data for one symbol
        # if data: st.dataframe(list(data.values())[0].head(), key='compare_data_head')


        # --- Run Backtests for Each Selected Strategy ---
        st.info("Running backtests for selected strategies...")
        st.toast("Running backtests...", icon="⚙️")
        backtest_results = {}
        raw_equity_curves = {} # To store potentially non-Series equity data initially
        all_trades = {} # To store trades for all strategies

        # Create a progress bar for backtests
        backtest_progress_bar = st.progress(0)
        backtest_status_text = st.empty()

        for i, strategy_name in enumerate(selected_strategy_names):
            backtest_status_text.text(f"Running backtest for {strategy_name} ({i+1}/{len(selected_strategy_names)})...")
            backtest_progress_bar.progress((i + 1) / len(selected_strategy_names))

            StrategyClass = AVAILABLE_STRATEGIES.get(strategy_name)
            if StrategyClass:
                try:
                    # Instantiate the strategy with the CONFIGURED parameters
                    strategy_params_to_use = configured_strategy_params.get(strategy_name, {})
                    if "Error" in strategy_params_to_use:
                         st.warning(f"Skipping backtest for {strategy_name} due to parameter configuration error.")
                         st.toast(f"Skipping backtest for {strategy_name} due to config error.", icon="⚠️")
                         backtest_results[strategy_name] = {"Error": strategy_params_to_use["Error"]}
                         continue # Skip to the next strategy


                    strategy_instance = StrategyClass(**strategy_params_to_use)


                    # Run the multi-asset backtest
                    backtester = Backtester(initial_capital=initial_capital)
                    results = backtester.run_backtest(
                        data.copy(), # Use a copy of the data for each backtest
                        strategy_instance,
                        stop_loss_pct=compare_stop_loss_pct,
                        take_profit_pct=compare_take_profit_pct,
                        position_sizing_method=compare_position_sizing_method,
                        position_sizing_value=compare_position_sizing_value
                    )

                    if "error" in results:
                        st.warning(f"Backtest failed for {strategy_name}: {results['error']}")
                        st.toast(f"Backtest failed for {strategy_name}.", icon="❌")
                        backtest_results[strategy_name] = {"Error": results['error']}
                    else:
                        st.toast(f"Backtest completed for {strategy_name}.", icon="✅")
                        # --- Extract metrics directly from the results dictionary ---
                        metrics = {
                            "total_return": results.get("total_return"),
                            "annualized_return": results.get("annualized_return"),
                            "sharpe_ratio": results.get("sharpe_ratio"),
                            "sortino_ratio": results.get("sortino_ratio"),
                            "calmar_ratio": results.get("calmar_ratio"),
                            "max_drawdown_pct": results.get("max_drawdown_pct"),
                            "max_drawdown_value": results.get("max_drawdown_value"),
                            "trade_count": results.get("trade_count"),
                        }
                        backtest_results[strategy_name] = metrics

                        # Store the raw portfolio history for later processing
                        if 'portfolio_history' in results and results['portfolio_history'] is not None and 'total' in results['portfolio_history']:
                             raw_equity_curves[strategy_name] = results['portfolio_history']['total']
                        else:
                             st.warning(f"No portfolio history found for {strategy_name}.")


                        # Store trades for later processing
                        if 'trades' in results and results['trades'] is not None:
                             all_trades[strategy_name] = results['trades']
                        else:
                             st.warning(f"No trades found for {strategy_name}.")


                except Exception as e:
                    st.warning(f"An error occurred during backtesting for {strategy_name}: {e}")
                    st.toast(f"Error during backtest for {strategy_name}: {e}", icon="❌")
                    backtest_results[strategy_name] = {"Error": str(e)}

        backtest_status_text.text("Backtests finished.")
        backtest_progress_bar.progress(1.0) # Ensure progress bar is full
        st.toast("All backtests finished!", icon="✅")

        # Store results in session state for saving
        st.session_state['last_compare_backtest_results'] = {
            'metrics': backtest_results,
            'equity_curves': raw_equity_curves,
            'trades': all_trades,
            'initial_capital': initial_capital # Store initial capital for plotting/saving
        }


        # --- Display Comparative Metrics ---
        st.subheader("Comparative Metrics")
        if backtest_results:
            metrics_list = []
            for strategy_name, metrics in backtest_results.items():
                # Add strategy name as a column
                metrics_data = {"Strategy": strategy_name}
                metrics_data.update(metrics)
                metrics_list.append(metrics_data)

            if metrics_list:
                 metrics_df = pd.DataFrame(metrics_list)
                 # Set Strategy column as index for better display
                 metrics_df.set_index("Strategy", inplace=True)

                 # Format metric columns for display
                 for col in ["total_return", "annualized_return", "max_drawdown_pct"]:
                      if col in metrics_df.columns:
                           metrics_df[col] = metrics_df[col].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')

                 for col in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
                      if col in metrics_df.columns:
                           metrics_df[col] = metrics_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

                 # Format max_drawdown_value and trade_count as integers if they exist
                 for col in ["max_drawdown_value", "trade_count"]:
                      if col in metrics_df.columns:
                           metrics_df[col] = metrics_df[col].apply(lambda x: f'{int(x):,}' if pd.notna(x) and pd.api.types.is_numeric_dtype(type(x)) else 'N/A')


                 st.dataframe(metrics_df, key='comparative_metrics_table')
            else:
                 st.info("No backtest metrics available to display.")

        else:
            st.warning("No backtest results were successfully generated.")


        # --- Process and Plot Comparative Equity Curves using the interactive utility function ---
        st.subheader("Comparative Equity Curves")

        # Filter and process raw equity curves to ensure they are valid Series with DatetimeIndex
        valid_equity_curves = {}
        for strategy_name, raw_curve_data in raw_equity_curves.items():
             equity_curve = None # Initialize to None for each strategy

             if isinstance(raw_curve_data, pd.Series) and not raw_curve_data.empty:
                  equity_curve = raw_curve_data.copy() # Work on a copy
                  if not isinstance(equity_curve.index, pd.DatetimeIndex):
                       try:
                            equity_curve.index = pd.to_datetime(equity_curve.index)
                       except Exception as e:
                            st.warning(f"Could not convert index to DatetimeIndex for {strategy_name} equity curve: {e}")
                            equity_curve = None # Discard if conversion fails
                  # Ensure the index is sorted for plotting
                  if equity_curve is not None and not equity_curve.index.is_monotonic_increasing:
                       equity_curve = equity_curve.sort_index()

             if equity_curve is not None:
                  valid_equity_curves[strategy_name] = equity_curve
             else:
                  st.warning(f"No valid equity curve generated for {strategy_name} for plotting.")


        if valid_equity_curves:
            # Use the modified plot_equity_curve utility function (now interactive and handles dict)
            try:
                fig = plot_equity_curve(valid_equity_curves, initial_capital=initial_capital)
                if fig: # Check if the plotting function returned a figure
                    st.plotly_chart(fig) # Use st.plotly_chart for interactive plots
                else:
                    st.warning("Could not generate interactive equity curve plot.")
            except Exception as e:
                st.error(f"An error occurred while plotting equity curves: {e}")
        else:
            if raw_equity_curves: # If raw_equity_curves dict is not empty but valid_equity_curves is
                 st.warning("None of the generated equity curves were valid for plotting.")
            else: # If the raw_equity_curves dict was empty from the start
                 st.info("No equity curves available to plot.")

        # --- Display Trades (Optional - can be added here or in a separate expander) ---
        # st.subheader("Trades")
        # if all_trades:
        #      # You would need to format and display all_trades here, perhaps in a table
        #      st.json(all_trades) # Example: Display as JSON, you might want a DataFrame

        # else:
        #      st.info("No trades recorded for the selected strategies.")


# --- Save Results Button ---
# Display the save button only if there are backtest results in session state
if 'last_compare_backtest_results' in st.session_state and st.session_state['last_compare_backtest_results'] is not None:
    results_to_save = st.session_state['last_compare_backtest_results']

    # Prepare data for download
    # Comparative metrics
    metrics_data = results_to_save.get('metrics', {})
    metrics_json = json.dumps(metrics_data, indent=4)

    # Equity curves (save each as a separate CSV or combine into one)
    equity_curves_data = results_to_save.get('equity_curves', {})
    # Trades (save each as a separate CSV or combine)
    trades_data = results_to_save.get('trades', {})

    # Use the placeholder to render the download buttons
    with save_button_placeholder.container():
        st.markdown("---") # Add a separator before save buttons
        st.subheader("Download Results")
        col_metrics, col_equity, col_trades = st.columns(3)

        with col_metrics:
            st.download_button(
                label="Metrics (JSON)",
                data=metrics_json,
                file_name=f"comparative_metrics_{datetime.date.today().strftime('%Y%m%d')}.json",
                mime="application/json",
                key='download_compare_metrics_json',
                on_click=lambda: st.toast("Comparative Metrics JSON prepared for download!", icon="💾")
            )

        with col_equity:
            if equity_curves_data:
                # Option 1: Save each equity curve as a separate CSV
                # This requires creating multiple download buttons or a zip file.
                # For simplicity, let's create a single CSV with all equity curves (requires aligning indices)
                combined_equity_df = pd.DataFrame()
                for strategy_name, equity_curve in equity_curves_data.items():
                     if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
                          # Ensure index is DatetimeIndex and sorted
                          if not isinstance(equity_curve.index, pd.DatetimeIndex):
                               try:
                                    equity_curve.index = pd.to_datetime(equity_curve.index)
                               except Exception as e:
                                    st.warning(f"Could not process equity curve for {strategy_name} for combined CSV: {e}")
                                    continue # Skip this curve if index conversion fails
                          if not equity_curve.index.is_monotonic_increasing:
                               equity_curve = equity_curve.sort_index()

                          # Rename the series to the strategy name for the combined DataFrame
                          equity_curve.name = strategy_name
                          if combined_equity_df.empty:
                               combined_equity_df = equity_curve.to_frame()
                          else:
                               combined_equity_df = combined_equity_df.join(equity_curve, how='outer') # Use outer join to keep all dates

                if not combined_equity_df.empty:
                     # Ensure Date column is added and formatted
                     combined_equity_df.reset_index(inplace=True)
                     combined_equity_df.rename(columns={'index': 'Date'}, inplace=True)
                     combined_equity_df['Date'] = combined_equity_df['Date'].dt.strftime('%Y-%m-%d')

                     st.download_button(
                        label="Equity Curves (CSV)",
                        data=combined_equity_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"comparative_equity_curves_{datetime.date.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key='download_compare_equity_csv',
                        on_click=lambda: st.toast("Comparative Equity Curves CSV prepared for download!", icon="💾")
                    )
                else:
                     st.info("No valid equity curve data to download.")
            else:
                 st.info("No equity curve data to download.")


        with col_trades:
            if trades_data:
                # Option 1: Save all trades combined into a single CSV
                combined_trades_list = []
                for strategy_name, trades_list in trades_data.items():
                     if isinstance(trades_list, list):
                          for trade in trades_list:
                               # Add strategy name to each trade record
                               trade_with_strategy = trade.copy()
                               trade_with_strategy['strategy'] = strategy_name
                               combined_trades_list.append(trade_with_strategy)

                if combined_trades_list:
                     try:
                          combined_trades_df = pd.DataFrame(combined_trades_list)
                          # Convert timestamp columns to datetime if they exist
                          for col in ['entry_time', 'exit_time']: # Adjust column names as per your trades structure
                               if col in combined_trades_df.columns:
                                    combined_trades_df[col] = pd.to_datetime(combined_trades_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

                          st.download_button(
                                label="All Trades (CSV)",
                                data=combined_trades_df.to_csv(index=False).encode('utf-8'),
                                file_name=f"comparative_trades_{datetime.date.today().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                key='download_compare_trades_csv',
                                on_click=lambda: st.toast("Comparative Trades CSV prepared for download!", icon="💾")
                            )
                     except Exception as e:
                          st.warning(f"Could not process trades data for combined CSV: {e}")
                          st.info("Trades data could not be processed for download.")

                else:
                     st.info("No trades data to download.")
            else:
                 st.info("No trades data to download.")
