import streamlit as st
import pandas as pd
import datetime
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff # For distribution plots
import plotly.express as px # For box plots or histograms
import plotly.graph_objects as go # For equity curve plot
from streamlit_searchbox import st_searchbox
import random # For randomization in simulations
import time # Import time for progress tracking

# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from data_sources.base_data_source import BaseDataSource
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource
from strategies.base_strategy import BaseStrategy
from backtester.backtester import Backtester # We will use the Backtester


# --- Helper function for symbol searchbox calling the data source method ---
def search_symbols_montecarlo(search_term: str) -> list[str]:
    """
    Calls the search_symbols method of the currently selected data source for the Monte Carlo page.
    Accesses AVAILABLE_DATA_SOURCES from Streamlit's session state.
    """
    # Ensure AVAILABLE_DATA_SOURCES is in session state
    AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
    if not AVAILABLE_DATA_SOURCES:
        # This should ideally not happen if app.py runs first, but as a safeguard
        print("Error: AVAILABLE_DATA_SOURCES not found in session state.")
        return ["Error: Data sources not loaded."]

    selected_data_source_name = st.session_state.get('selected_data_source_name', list(st.session_state.get('available_data_sources', {}).keys())[0])
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


# --- UI Elements and Logic for the Monte Carlo Simulation Page ---
# This code will be executed directly by Streamlit when the page is selected.

st.title("Monte Carlo Simulation")

# Retrieve available strategies and data sources from session state
AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})
AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])


if not AVAILABLE_STRATEGIES:
    st.warning("No trading strategies available for simulation.")
    st.stop()


st.header("Simulation Configuration")

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
        start_date = st.date_input("Simulation Start Date", today - datetime.timedelta(days=365*3), key='mc_start_date')
    with col_date_end:
        end_date = st.date_input("Simulation End Date", today, key='mc_end_date')

    st.subheader("Symbol Selection")
    st.write("Select symbols for multi-asset simulation:")
    num_symbols = st.number_input("Number of Symbols", min_value=1, value=1, step=1, key='mc_num_symbols')
    selected_symbols = []
    for i in range(num_symbols):
        symbol = st_searchbox(
            search_symbols_montecarlo,
            label=f"Search and Select Symbol {i+1}",
            key=f"mc_symbol_searchbox_{i}"
        )
        if symbol:
            selected_symbols.append(symbol)

    st.write(f"Selected Symbols: **{', '.join(selected_symbols) if selected_symbols else 'None'}**")


# Use a container for Strategy and Fixed Parameters
with st.container(border=True):
    st.subheader("Strategy and Fixed Parameters")

    selected_strategy_name = st.selectbox(
        "Select Strategy for Simulation",
        list(AVAILABLE_STRATEGIES.keys()),
        key='mc_strategy_select'
    )
    StrategyClass = AVAILABLE_STRATEGIES.get(selected_strategy_name)

    # Use an expander for Fixed Parameters
    with st.expander("Configure Fixed Strategy Parameters"):
        strategy_params = {}
        if StrategyClass:
            if hasattr(StrategyClass, 'get_params') and callable(getattr(StrategyClass, 'get_params')):
                try:
                    temp_strategy = StrategyClass()
                    default_params = temp_strategy.get_params()
                    if default_params:
                        param_cols = st.columns(min(len(default_params), 3))
                        param_col_index = 0
                        for param_name, default_value in default_params.items():
                            current_col = param_cols[param_col_index % len(param_cols)]
                            with current_col:
                                # Use fixed value inputs for parameters
                                current_value = st.session_state.get(f'mc_param_{param_name}', default_value) # Get from session state
                                if isinstance(default_value, int):
                                     strategy_params[param_name] = st.number_input(f"Fixed Value ({param_name})", value=int(current_value), step=1, key=f'mc_param_{param_name}_int')
                                elif isinstance(default_value, float):
                                     strategy_params[param_name] = st.number_input(f"Fixed Value ({param_name})", value=float(current_value), step=0.1, format="%.2f", key=f'mc_param_{param_name}_float')
                                else:
                                     strategy_params[param_name] = st.text_input(f"Fixed Value ({param_name})", value=str(current_value), key=f'mc_param_{param_name}_text')

                                # Store fixed value in session state
                                st.session_state[f'mc_param_{param_name}'] = strategy_params[param_name]

                    else:
                        st.info("This strategy has no configurable parameters.")

                except Exception as e:
                    st.warning(f"Could not load parameters for {selected_strategy_name}: {e}")

# Use a container for Monte Carlo Specific Parameters
with st.container(border=True):
    st.subheader("Monte Carlo Parameters")
    n_simulations = st.number_input("Number of Simulations", min_value=10, value=100, step=10, key='mc_n_simulations')
    price_noise_pct = st.number_input("Price Noise (%) (Adds random variation to prices)", min_value=0.0, value=0.5, step=0.1, format="%.2f", key='mc_price_noise_pct') / 100.0


# Use a container for Initial Capital, Risk Management, and Position Sizing
with st.container(border=True):
    st.subheader("Capital, Risk, and Position Management")
    initial_capital = st.number_input("Initial Capital for Simulations", min_value=1000.0, value=100000.0, step=1000.0, key='mc_initial_capital')

    col_sl, col_tp = st.columns(2)
    with col_sl:
        mc_stop_loss_pct = st.number_input("Global Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.2f", key='mc_stop_loss_pct') / 100.0
    with col_tp:
        mc_take_profit_pct = st.number_input("Global Take Profit (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key='mc_take_profit_pct') / 100.0

    st.subheader("Position Sizing")
    mc_position_sizing_method = st.selectbox(
        "Method",
        ['Full Equity', 'Fixed Amount', 'Percentage of Equity'],
        key='mc_position_sizing_method'
    )
    mc_position_sizing_value = 0.0
    if mc_position_sizing_method == 'Fixed Amount':
        mc_position_sizing_value = st.number_input("Amount ($)", min_value=1.0, value=1000.0, step=100.0, key='mc_position_sizing_amount')
    elif mc_position_sizing_method == 'Percentage of Equity':
        mc_position_sizing_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f", key='mc_position_sizing_percentage') / 100.0


# --- Run Simulation Button ---
is_disabled = not selected_symbols or not selected_strategy_name or initial_capital <= 0 or n_simulations <= 0
if mc_position_sizing_method != 'Full Equity' and mc_position_sizing_value <= 0:
     is_disabled = True

st.markdown("---") # Add a separator before the button
if st.button("Run Monte Carlo Simulation", key='run_mc_simulation_button', disabled=is_disabled):
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
    elif not selected_symbols:
         st.warning("Please select at least one symbol for simulation.")
    elif not selected_strategy_name:
         st.warning("Please select a strategy.")
    elif initial_capital <= 0:
         st.warning("Initial capital must be greater than 0.")
    elif n_simulations <= 0:
         st.warning("Number of simulations must be greater than 0.")
    elif mc_position_sizing_method != 'Full Equity' and mc_position_sizing_value <= 0:
         st.warning(f"Please enter a valid {mc_position_sizing_method} value.")
    else:
        st.subheader("Simulation Results")
        if selected_data_source_name == 'Alpha Vantage':
            alphavantage_key = st.session_state.get('alphavantage_api_key')
            if not alphavantage_key:
                st.error("Alpha Vantage API key is not set in Settings.")
                st.toast("Alpha Vantage API key is not set.", icon="❌")
                st.stop() # Stop execution if key is missing for Alpha Vantage
            data_source_instance = DataSourceClass(api_key=alphavantage_key)
        else:
                    # Instantiate other data sources without the API key parameter
            data_source_instance = DataSourceClass()

                    # --- Fetch Data ---
        st.info(f"Fetching data for {', '.join(selected_symbols)} from {start_date} to {end_date} using {selected_data_source_name}...")
        data = {}
        fetch_errors = []
        for symbol in selected_symbols:
            try:
                asset_data = data_source_instance.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if not asset_data.empty:
                    data[symbol] = asset_data
                else:
                    fetch_errors.append(f"No data fetched for {symbol}")
            except Exception as e:
                fetch_errors.append(f"Error fetching data for {symbol}: {e}")

        if fetch_errors:
            for error in fetch_errors:
                st.error(error)
            if not data:
                 st.error("No data available for simulation after fetching.")
                 st.stop()

        if not data:
             st.error("No data available for simulation.")
             st.stop()

        st.success(f"Data fetched successfully for {list(data.keys())}.")
        # Optional: Display head of data for one symbol
        # if data: st.dataframe(list(data.values())[0].head(), key='mc_data_head')


        # --- Run Monte Carlo Simulations ---
        st.info(f"Running {n_simulations} Monte Carlo simulations...")

        simulation_metrics = {
            "Total Return": [],
            "Annualized Return": [],
            "Sharpe Ratio": [],
            "Sortino Ratio": [],
            "Calmar Ratio": [],
            "Max Drawdown (%)": [],
            "Trade Count": []
        }
        equity_curves_simulations = [] # Store equity curves for plotting

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        for i in range(n_simulations):
            status_text.text(f"Running simulation {i + 1}/{n_simulations}...")

            # --- Randomize Data (Example: Add Price Noise) ---
            # Create a noisy version of the data for this simulation
            noisy_data = {}
            for symbol, df in data.items():
                 noisy_df = df.copy()
                 # Add random noise to Close price (example)
                 noise = np.random.normal(0, price_noise_pct * noisy_df['Close'].mean(), size=len(noisy_df))
                 noisy_df['Close'] = noisy_df['Close'] + noise
                 # Ensure prices don't go below zero
                 noisy_df['Close'] = noisy_df['Close'].clip(lower=0.01) # Keep a small positive value

                 # Simple way to adjust OHLC based on noisy close - could be more sophisticated
                 # Apply noise proportionally to the difference from the original close
                 original_close = df['Close']
                 # Avoid division by zero or inf, handle potential missing values before calculation
                 noise_factor = (noisy_df['Close'] / original_close.replace(0, np.nan)).replace([np.inf, -np.inf], 1).fillna(1)

                 noisy_df['Open'] = original_close * noise_factor # Apply noise factor to open
                 noisy_df['High'] = original_close * noise_factor # Apply noise factor to high
                 noisy_df['Low'] = original_close * noise_factor # Apply noise factor to low

                 # Re-calculate high and low based on noisy OHLC
                 noisy_df['High'] = noisy_df[['Open', 'High', 'Close']].max(axis=1)
                 noisy_df['Low'] = noisy_df[['Open', 'Low', 'Close']].min(axis=1)

                 noisy_data[symbol] = noisy_df


            # --- Run Backtest with Randomized Data ---
            try:
                # Instantiate the strategy with the fixed parameters
                strategy_instance = StrategyClass(**strategy_params)

                backtester = Backtester(initial_capital=initial_capital)
                results = backtester.run_backtest(
                    noisy_data, # Use the noisy data
                    strategy_instance,
                    stop_loss_pct=mc_stop_loss_pct,
                    take_profit_pct=mc_take_profit_pct,
                    position_sizing_method=mc_position_sizing_method,
                    position_sizing_value=mc_position_sizing_value
                )

                if "error" in results:
                    st.warning(f"Backtest failed in simulation {i + 1}: {results['error']}")
                else:
                    # Store metrics from the simulation
                    simulation_metrics["Total Return"].append(results.get('total_return'))
                    simulation_metrics["Annualized Return"].append(results.get('annualized_return'))
                    simulation_metrics["Sharpe Ratio"].append(results.get('sharpe_ratio'))
                    simulation_metrics["Sortino Ratio"].append(results.get('sortino_ratio'))
                    simulation_metrics["Calmar Ratio"].append(results.get('calmar_ratio'))
                    simulation_metrics["Max Drawdown (%)"].append(results.get('max_drawdown_pct'))
                    simulation_metrics["Trade Count"].append(results.get('trade_count'))

                    # Store equity curve
                    if 'portfolio_history' in results:
                         equity_curves_simulations.append(results['portfolio_history']['total'])


            except Exception as e:
                st.error(f"Error during backtest in simulation {i + 1}: {e}")

            # Update progress bar
            progress_bar.progress((i + 1) / n_simulations)

        end_time = time.time()
        status_text.text(f"Monte Carlo Simulation finished in {end_time - start_time:.2f} seconds.")

        # --- Display Simulation Results ---
        if simulation_metrics["Total Return"]: # Check if any simulations were successful
            st.subheader("Simulation Results Summary")

            # Create a DataFrame from collected metrics
            metrics_df = pd.DataFrame(simulation_metrics)

            # Display summary statistics
            st.write("Summary Statistics:")
            st.dataframe(metrics_df.describe().T, key='mc_metrics_summary')

            # Display distribution plots for key metrics
            st.subheader("Distribution of Metrics")

            # Example: Histogram for Total Return
            if not metrics_df["Total Return"].isnull().all(): # Check if column has valid data
                 fig_ret = px.histogram(metrics_df, x="Total Return", nbins=50, title="Distribution of Total Return")
                 st.plotly_chart(fig_ret)

            # Example: Box plot for Sharpe Ratio
            if not metrics_df["Sharpe Ratio"].isnull().all():
                 fig_sharpe = px.box(metrics_df, y="Sharpe Ratio", title="Distribution of Sharpe Ratio")
                 st.plotly_chart(fig_sharpe)

            # Add plots for other metrics as desired

            st.subheader("Simulated Equity Curves")
            if equity_curves_simulations:
                 fig_equity = go.Figure()
                 for j, equity_curve in enumerate(equity_curves_simulations):
                      fig_equity.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name=f'Simulation {j+1}'))

                 fig_equity.update_layout(
                     title='Simulated Portfolio Equity Curves',
                     xaxis_title='Date',
                     yaxis_title='Portfolio Value'
                 )
                 st.plotly_chart(fig_equity)
            else:
                 st.warning("No equity curves available from simulations.")


        else:
            st.warning("No successful simulations were completed.")


# Display warnings if button is disabled
elif not selected_symbols:
     st.warning("Please select at least one symbol for simulation.")
elif not selected_strategy_name:
     st.warning("Please select a strategy.")
elif initial_capital <= 0:
     st.warning("Initial capital must be greater than 0.")
elif n_simulations <= 0:
     st.warning("Number of simulations must be greater than 0.")
elif mc_position_sizing_method != 'Full Equity' and mc_position_sizing_value <= 0:
     st.warning(f"Please enter a valid {mc_position_sizing_method} value.")

