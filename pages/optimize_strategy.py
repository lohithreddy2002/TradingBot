import streamlit as st
import pandas as pd
import sys
import os
import time # Import time for sleep
import optuna # Import Optuna
import plotly.graph_objects as go # Import Plotly for Optuna visualizations
import numpy as np # Import numpy for sensitivity analysis
import itertools # Import itertools for sensitivity analysis combinations
import datetime # Import datetime for date inputs
import json # Import json for saving results


# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules (strategies, data sources, backtester)
from strategies.base_strategy import BaseStrategy
from backtester.backtester import Backtester
from data_sources.base_data_source import BaseDataSource
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource
from streamlit_searchbox import st_searchbox

# Initialize stop flags in session state if they don't exist
if 'stop_optuna' not in st.session_state:
    st.session_state.stop_optuna = False
if 'stop_walkforward' not in st.session_state:
    st.session_state.stop_walkforward = False
if 'stop_sensitivity' not in st.session_state:
    st.session_state.stop_sensitivity = False

# Initialize running flags in session state if they don't exist
if 'is_optuna_optimization_running' not in st.session_state:
    st.session_state.is_optuna_optimization_running = False
if 'is_walk_forward_testing_running' not in st.session_state:
    st.session_state.is_walk_forward_testing_running = False
if 'is_parameter_sensitivity_analysis_running' not in st.session_state:
    st.session_state.is_parameter_sensitivity_analysis_running = False

# Initialize results storage in session state
if 'last_analysis_results' not in st.session_state:
    st.session_state['last_analysis_results'] = None


# --- Helper function for symbol searchbox calling the data source method ---
def search_symbols_optimize(search_term: str) -> list[str]:
    """
    Calls the search_symbols method of the currently selected data source for the optimization page.
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


# --- Optuna Objective Function ---
# This objective function needs access to the parameter ranges defined in the UI
# We will pass them when creating the study
def optuna_objective(trial, data, StrategyClass, initial_capital, stop_loss_pct, take_profit_pct,
                     position_sizing_method, position_sizing_value, optimization_metric, parameter_configs_for_trial):
    """
    Optuna objective function to minimize or maximize a performance metric.
    Runs a backtest with parameters suggested by the trial, using provided parameter configurations.
    parameter_configs_for_trial is a dictionary where values are dicts like {'start': ..., 'end': ..., 'step': ..., 'type': ...}
    for parameters to be optimized, or {'fixed_value': ..., 'type': ...} for fixed parameters.
    """
    # Check stop flag at the beginning of each trial
    if st.session_state.stop_optuna:
        raise optuna.exceptions.TrialPruned("Optimization stopped by user.")


    strategy_params = {}
    # Suggest or use fixed parameters based on the provided configurations
    for param_name, param_config in parameter_configs_for_trial.items():
        param_type = param_config.get('type') # Use .get for safety

        try:
            if 'start' in param_config and 'end' in param_config and 'step' in param_config and param_type is not None:
                 # Parameter is configured with a range for optimization
                 start_value = param_config['start']
                 end_value = param_config['end']
                 step_value = param_config['step']

                 if param_type == int:
                     low = int(start_value)
                     high = int(end_value)
                     if low > high: low, high = high, low
                     strategy_params[param_name] = trial.suggest_int(param_name, low, high, step=int(step_value))
                 elif param_type == float:
                     low = float(start_value)
                     high = float(end_value)
                     if low > high: low, high = high, low
                     strategy_params[param_name] = trial.suggest_float(param_name, low, high, step=float(step_value))
                 else:
                      # Should not happen if filtering is correct, but as a safeguard
                      print(f"Warning: Parameter '{param_name}' has a non-numeric type and cannot be optimized.")
                      return -np.inf # Return a poor value

            elif 'fixed_value' in param_config and param_type is not None:
                 # Parameter is configured with a fixed value
                 strategy_params[param_name] = param_config['fixed_value']
            else:
                 # Invalid parameter configuration
                 print(f"Error: Invalid parameter configuration for '{param_name}'.")
                 return -np.inf # Return a poor value

        except Exception as e:
            print(f"Error processing parameter '{param_name}' in trial {trial.number}: {e}")
            return -np.inf # Return a poor value


    # Instantiate the strategy with the suggested/fixed parameters
    try:
        strategy_instance = StrategyClass(**strategy_params)
    except Exception as e:
        # If instantiation fails, return a poor metric value
        print(f"Error initializing strategy in trial {trial.number} with params {strategy_params}: {e}")
        return -np.inf # For maximization, return negative infinity

    # Run the backtest
    backtester = Backtester(initial_capital=initial_capital)
    try:
        results = backtester.run_backtest(
            data.copy(), # Use a copy of the data
            strategy_instance,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            position_sizing_method=position_sizing_method,
            position_sizing_value=position_sizing_value
        )

        if "error" in results:
            print(f"Backtest failed in trial {trial.number} with params {strategy_params}: {results['error']}")
            return -np.inf # For maximization, return negative infinity

        # Get the value of the optimization metric
        # Convert metric name to result key (e.g., "Sharpe Ratio" -> "sharpe_ratio")
        metric_key = optimization_metric.replace(' (%)', '').replace(' ', '_').lower()
        metric_value = results.get(metric_key)

        # Handle metrics where lower is better (e.g., Max Drawdown) by negating
        if optimization_metric == "Max Drawdown (%)":
             return -metric_value if metric_value is not None else -np.inf # Maximize negative drawdown (minimize drawdown)
        else:
             return metric_value if metric_value is not None else -np.inf # Maximize other metrics

    except Exception as e:
        print(f"Error during backtest in trial {trial.number} with params {strategy_params}: {e}")
        return -np.inf # For maximization, return negative infinity


# --- UI Elements and Logic for the Strategy Analysis Page ---
# This code will be executed directly by Streamlit when the page is selected.

st.title("Strategy Analysis")

# Retrieve available strategies and data sources from session state
AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})
AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])


if not AVAILABLE_STRATEGIES:
    st.warning("No trading strategies available for analysis.")
    st.stop()


st.header("Analysis Configuration")

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
        start_date = st.date_input("Analysis Start Date", today - datetime.timedelta(days=365*5), key='analysis_start_date')
    with col_date_end:
        end_date = st.date_input("Analysis End Date", today, key='analysis_end_date')

    st.subheader("Symbol Selection")
    st.write("Select symbols for multi-asset analysis:")
    num_symbols = st.number_input("Number of Symbols", min_value=1, value=1, step=1, key='analysis_num_symbols')
    selected_symbols = []
    for i in range(num_symbols):
        symbol = st_searchbox(
            search_symbols_optimize, # Reuse the optimize search function
            label=f"Search and Select Symbol {i+1}",
            key=f"analysis_symbol_searchbox_{i}"
        )
        if symbol:
            selected_symbols.append(symbol)

    st.write(f"Selected Symbols: **{', '.join(selected_symbols) if selected_symbols else 'None'}**")


# Use a container for Analysis Mode and Strategy Selection
with st.container(border=True):
    st.subheader("Analysis Mode and Strategy")

    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["Optuna Optimization", "Walk-Forward Testing", "Parameter Sensitivity Analysis"],
        key='analysis_mode_select'
    )

    selected_strategy_name = st.selectbox(
        "Select Strategy",
        list(AVAILABLE_STRATEGIES.keys()),
        key='analysis_strategy_select'
    )
    StrategyClass = AVAILABLE_STRATEGIES.get(selected_strategy_name)

    # Store parameter configuration (ranges for Optuna/Sensitivity, fixed for Walk-Forward) in session state
    if 'analysis_parameter_config' not in st.session_state:
        st.session_state['analysis_parameter_config'] = {}

    parameter_config_ui = {} # Dictionary to hold parameter config defined in the UI

    if StrategyClass:
        st.write(f"Configure Parameters for {selected_strategy_name}:")
        if hasattr(StrategyClass, 'get_params') and callable(getattr(StrategyClass, 'get_params')):
            try:
                temp_strategy = StrategyClass()
                default_params = temp_strategy.get_params()
                if default_params:
                    for param_name, default_value in default_params.items():
                        st.write(f"Configure **{param_name}** (Default: {default_value})")

                        # UI for parameter ranges (used by Optuna, Walk-Forward Training, and Sensitivity)
                        col_start, col_end, col_step = st.columns(3)
                        with col_start:
                            current_start = st.session_state['analysis_parameter_config'].get(param_name, {}).get('start', default_value * 0.8 if isinstance(default_value, (int, float)) else default_value)
                            if isinstance(default_value, int):
                                start_value = st.number_input(f"Start ({param_name})", value=int(current_start), step=1, key=f'analysis_param_{param_name}_start_int')
                            elif isinstance(default_value, float):
                                start_value = st.number_input(f"Start ({param_name})", value=float(current_start), step=0.1, format="%.2f", key=f'analysis_param_{param_name}_start_float')
                            else:
                                start_value = st.text_input(f"Start ({param_name})", value=str(current_start), key=f'analysis_param_{param_name}_start_text')

                        with col_end:
                            current_end = st.session_state['analysis_parameter_config'].get(param_name, {}).get('end', default_value * 1.2 if isinstance(default_value, (int, float)) else default_value)
                            if isinstance(default_value, int):
                                end_value = st.number_input(f"End ({param_name})", value=int(current_end), step=1, key=f'analysis_param_{param_name}_end_int')
                            elif isinstance(default_value, float):
                                end_value = st.number_input(f"End ({param_name})", value=float(current_end), step=0.1, format="%.2f", key=f'analysis_param_{param_name}_end_float')
                            else:
                                end_value = st.text_input(f"End ({param_name})", value=str(current_end), key=f'analysis_param_{param_name}_end_text')

                        with col_step:
                            current_step = st.session_state['analysis_parameter_config'].get(param_name, {}).get('step', 1 if isinstance(default_value, int) else (0.1 if isinstance(default_value, float) else None))
                            if isinstance(default_value, int):
                                step_value = st.number_input(f"Step ({param_name})", min_value=1, value=int(current_step), step=1, key=f'analysis_param_{param_name}_step_int')
                            elif isinstance(default_value, float):
                                step_value = st.number_input(f"Step ({param_name})", min_value=0.01, value=float(current_step), step=0.01, format="%.2f", key=f'analysis_param_{param_name}_step_float')
                            else:
                                step_value = None # Step not applicable for text


                        # Store the range configuration
                        parameter_config_ui[param_name] = {
                            'start': start_value,
                            'end': end_value,
                            'step': step_value,
                            'type': type(default_value)
                        }
                        st.session_state['analysis_parameter_config'][param_name] = parameter_config_ui[param_name]

                        # For Sensitivity Analysis, also allow selecting which parameters to make sensitive
                        if analysis_mode == "Parameter Sensitivity Analysis":
                             if 'sensitivity_params' not in st.session_state:
                                  st.session_state['sensitivity_params'] = []

                             is_sensitivity_param = st.checkbox(f"Include {param_name} in Sensitivity Analysis", key=f'sensitivity_checkbox_{param_name}', value=param_name in st.session_state['sensitivity_params'])
                             if is_sensitivity_param and param_name not in st.session_state['sensitivity_params']:
                                  st.session_state['sensitivity_params'].append(param_name)
                             elif not is_sensitivity_param and param_name in st.session_state['sensitivity_params']:
                                  st.session_state['sensitivity_params'].remove(param_name)
                             # Note: The range defined above will be used for the sensitive parameters.
                             # Non-sensitive parameters will use their 'start' value as a fixed value in Sensitivity Analysis.


                else:
                    st.info("This strategy has no configurable parameters.")

            except Exception as e:
                st.warning(f"Could not load parameters for {selected_strategy_name}: {e}")

# Use a container for Analysis Specific Parameters
with st.container(border=True):
    st.subheader(f"{analysis_mode} Parameters")

    if analysis_mode == "Optuna Optimization":
        n_trials = st.number_input("Number of Trials", min_value=1, value=50, step=1, key='optuna_n_trials')
        # Add other Optuna parameters if needed (e.g., timeout, sampler, pruner)

    elif analysis_mode == "Walk-Forward Testing":
        col_train, col_test = st.columns(2)
        with col_train:
            training_window_size = st.number_input("Training Window Size (Days)", min_value=30, value=365, step=30, key='wf_training_window')
        with col_test:
            testing_window_size = st.number_input("Testing Window Size (Days)", min_value=30, value=180, step=30, key='wf_testing_window')
        step_size = st.number_input("Step Size (Days)", min_value=1, value=30, step=1, key='wf_step_size')
        wf_optuna_trials = st.number_input("Optuna Trials per Training Window", min_value=1, value=20, step=1, key='wf_optuna_trials')
        st.info("Parameters configured above will be optimized within each training window.")


    elif analysis_mode == "Parameter Sensitivity Analysis":
        sensitivity_params = st.session_state.get('sensitivity_params', [])
        if not sensitivity_params:
             st.warning("Select at least one parameter for sensitivity analysis in the Strategy Configuration section.")
        elif len(sensitivity_params) > 2:
             st.warning("Sensitivity analysis visualization is best with 1 or 2 parameters. You have selected more.")
        st.info("Parameters selected for Sensitivity Analysis will be varied across their defined ranges. Other parameters will use their 'Start' value as a fixed value.")


# Use a container for Initial Capital, Risk Management, and Position Sizing
with st.container(border=True):
    st.subheader("Capital, Risk, and Position Management")
    # Get default initial capital from settings, allow override
    default_initial_capital = st.session_state.get('initial_capital', 100000.0)
    initial_capital = st.number_input("Initial Capital for Analysis Backtests", min_value=1000.0, value=default_initial_capital, step=1000.0, key='analysis_initial_capital_override')


    col_sl, col_tp = st.columns(2)
    with col_sl:
        analysis_stop_loss_pct = st.number_input("Global Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.2f", key='analysis_stop_loss_pct') / 100.0
    with col_tp:
        analysis_take_profit_pct = st.number_input("Global Take Profit (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key='analysis_take_profit_pct') / 100.0

    st.subheader("Position Sizing")
    analysis_position_sizing_method = st.selectbox(
        "Method",
        ['Full Equity', 'Fixed Amount', 'Percentage of Equity'],
        key='analysis_position_sizing_method'
    )
    analysis_position_sizing_value = 0.0
    if analysis_position_sizing_method == 'Fixed Amount':
        analysis_position_sizing_value = st.number_input("Amount ($)", min_value=1.0, value=1000.0, step=100.0, key='analysis_position_sizing_amount')
    elif analysis_position_sizing_method == 'Percentage of Equity':
        analysis_position_sizing_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f", key='analysis_position_sizing_percentage') / 100.0


# --- Optimization Metric ---
# This metric is used for Optuna and Walk-Forward optimization
# For Sensitivity, we might show multiple metrics
with st.container(border=True):
    st.subheader("Optimization/Analysis Metric")
    analysis_metric = st.selectbox(
        "Metric to Optimize/Analyze",
        ["Total Return", "Annualized Return", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown (%)"],
        key='analysis_metric_select'
    )


# --- Check conditions for disabling the button and display warnings ---
is_disabled = not selected_symbols or not selected_strategy_name or initial_capital <= 0

if analysis_mode == "Optuna Optimization":
     has_optuna_params = any('start' in config for config in parameter_config_ui.values())
     if not has_optuna_params:
          st.warning("Please configure at least one parameter range for Optuna Optimization.")
          is_disabled = True
     if n_trials <= 0:
          st.warning("Number of trials for Optuna must be greater than 0.")
          is_disabled = True
     if analysis_position_sizing_method != 'Full Equity' and analysis_position_sizing_value <= 0:
          st.warning(f"Please enter a valid {analysis_position_sizing_method} value.")
          is_disabled = True

elif analysis_mode == "Walk-Forward Testing":
     if training_window_size <= 0 or testing_window_size <= 0 or step_size <= 0 or wf_optuna_trials <= 0:
          st.warning("Please define valid window sizes and step size for Walk-Forward Testing.")
          is_disabled = True
     # Data availability check is done after fetching data.

elif analysis_mode == "Parameter Sensitivity Analysis":
     sensitivity_params = st.session_state.get('sensitivity_params', [])
     if not sensitivity_params:
          st.warning("Please select at least one parameter for sensitivity analysis in the Strategy Configuration section.")
          is_disabled = True
     elif len(sensitivity_params) > 2:
          st.warning("Sensitivity analysis visualization is best with 1 or 2 parameters. You have selected more.")
     # For sensitivity, we also need fixed values for non-sensitivity parameters
     # Check if all non-sensitivity parameters have a 'start' value defined to be used as fixed
     if StrategyClass and hasattr(StrategyClass, 'get_params') and callable(getattr(StrategyClass, 'get_params')):
          default_params = StrategyClass().get_params()
          for param_name in default_params.keys():
               if param_name not in sensitivity_params:
                    if param_name not in parameter_config_ui or 'start' not in parameter_config_ui[param_name]:
                         st.warning(f"Fixed value (Start) not defined for non-sensitivity parameter '{param_name}'.")
                         is_disabled = True
                         break # Exit loop once a missing fixed value is found
     if analysis_position_sizing_method != 'Full Equity' and analysis_position_sizing_value <= 0:
          st.warning(f"Please enter a valid {analysis_position_sizing_method} value.")
          is_disabled = True


# --- Run Analysis Button ---
st.markdown("---") # Add a separator before the button
# --- Debugging: Print statement before the button check ---
# print(f"Debug: 'Run Analysis' button disabled state: {is_disabled}")

# Use a column for the Run button and Stop button
col_run, col_stop = st.columns([1, 1])

# Determine if the stop button should be enabled
current_analysis_running_flag = st.session_state.get(f'is_{analysis_mode.replace(" ", "_").lower()}_running', False)

with col_run:
    run_button_clicked = st.button(f"Run {analysis_mode}", key='run_analysis_button', disabled=is_disabled or current_analysis_running_flag) # Disable run button while analysis is running

with col_stop:
    # Define stop button behavior based on analysis mode
    stop_button_key = f'stop_{analysis_mode.replace(" ", "_").lower()}_button'
    stop_button_clicked = st.button("Stop Analysis", key=stop_button_key, disabled=not current_analysis_running_flag)

# Add a placeholder for the Save Results button(s) for analysis
save_button_placeholder = st.empty()


# Logic to set stop flags when stop button is clicked
if stop_button_clicked:
    if analysis_mode == "Optuna Optimization":
        st.session_state.stop_optuna = True
        st.toast("Stopping Optuna Optimization...", icon="✋")
    elif analysis_mode == "Walk-Forward Testing":
        st.session_state.stop_walkforward = True
        st.toast("Stopping Walk-Forward Testing...", icon="✋")
    elif analysis_mode == "Parameter Sensitivity Analysis":
        st.session_state.stop_sensitivity = True
        st.toast("Stopping Sensitivity Analysis...", icon="✋")


if run_button_clicked:
    # Reset stop flags at the start of a new run
    st.session_state.stop_optuna = False
    st.session_state.stop_walkforward = False
    st.session_state.stop_sensitivity = False
    # Clear previous analysis results
    st.session_state['last_analysis_results'] = None


    # Set running flag for the current mode
    st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = True
    # Add a small delay to ensure the session state updates before the next rerun evaluates button states
    time.sleep(0.1)
    st.rerun() # Trigger a rerun to update button states


# The rest of the analysis logic should be outside the button click check
# so that it runs on each rerun while the analysis_mode_running flag is True

if st.session_state.get(f'is_{analysis_mode.replace(" ", "_").lower()}_running', False):
    # This block will now execute on reruns while the analysis is running

    # --- Debugging: Print statement inside the running block ---
    # print(f"Debug: Analysis mode '{analysis_mode}' is running.")


    # Re-check conditions inside the running block as a safeguard
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
        st.toast("Error: Start date must be before end date.", icon="❌")
        # print("Debug: Start date >= End date.")
        st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
        st.stop()
    elif not selected_symbols:
         st.warning("Please select at least one symbol for analysis.")
         st.toast("Please select at least one symbol for analysis.", icon="⚠️")
         # print("Debug: No symbols selected.")
         st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
         st.stop()
    elif not selected_strategy_name:
         st.warning("Please select a strategy.")
         st.toast("Please select a strategy.", icon="⚠️")
         # print("Debug: No strategy selected.")
         st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
         st.stop()
    elif initial_capital <= 0:
         st.warning("Initial capital must be greater than 0.")
         st.toast("Initial capital must be greater than 0.", icon="⚠️")
         # print("Debug: Initial capital <= 0.")
         st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
         st.stop()
    elif analysis_mode == "Optuna Optimization":
         has_optuna_params = any('start' in config for config in parameter_config_ui.values())
         if not has_optuna_params or n_trials <= 0:
              # Warning already displayed above, just stop
              # print("Debug: Optuna conditions not met inside running block.")
              st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
              st.stop()
     # Note: Walk-Forward and Sensitivity checks are done before the button is enabled,
     # but can be added here again for extra safety if needed.
    elif analysis_position_sizing_method != 'Full Equity' and analysis_position_sizing_value <= 0:
         # Warning already displayed above, just stop
         # print(f"Debug: Position sizing condition not met inside running block for {analysis_position_sizing_method}.")
         st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
         st.stop()



    st.subheader("Analysis Results")

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
            st.toast(f"Error fetching data for {symbol}: {e}", icon="❌")
        if not data:
                st.error("No data available for analysis after fetching.")
                st.toast("No data available for analysis.", icon="❌")
                # print("Debug: No data available after fetching.")
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                st.stop()

    if not data:
            st.error("No data available for analysis.")
            st.toast("No data available for analysis.", icon="❌")
            # print("Debug: No data available for analysis (initial check).")
            st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
            st.stop()

    st.success(f"Data fetched successfully for {list(data.keys())}.")
    st.toast("Data fetched successfully!", icon="✅")
    # Optional: Display head of data for one symbol
    # if data: st.dataframe(list(data.values())[0].head(), key='analysis_data_head')


    # --- Perform Selected Analysis ---

    if analysis_mode == "Optuna Optimization":
        st.info("Running Optuna optimization...")
        st.toast("Running Optuna optimization...", icon="⚙️")
        # print("Debug: Running Optuna Optimization.")

        # --- Filter parameter_config_ui for Optuna ---
        # Only include parameters with 'start', 'end', 'step', and 'type' keys
        parameter_ranges_for_optuna = {
            param_name: config for param_name, config in parameter_config_ui.items()
            if all(key in config for key in ['start', 'end', 'step', 'type'])
        }

        if not parameter_ranges_for_optuna:
                st.error("No valid parameter ranges configured for Optuna Optimization.")
                st.toast("No valid parameter ranges configured for Optuna.", icon="❌")
                # print("Debug: No valid parameter ranges for Optuna.")
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                st.stop()

        # Create a progress bar for Optuna
        optuna_progress_bar = st.progress(0)
        optuna_status_text = st.empty()


        # Define the objective function for Optuna
        def optuna_objective_with_ranges(trial):
                # Update progress bar inside the objective (called for each trial)
                optuna_status_text.text(f"Running Optuna Trial {trial.number + 1}/{n_trials}...")
                optuna_progress_bar.progress((trial.number + 1) / n_trials)
                time.sleep(0.05) # Add a small sleep to potentially help UI update
                # ... rest of objective
                return optuna_objective(trial, data, StrategyClass, initial_capital, analysis_stop_loss_pct,
                                analysis_take_profit_pct, analysis_position_sizing_method,
                                analysis_position_sizing_value, analysis_metric, parameter_ranges_for_optuna)


        # Create an Optuna study
        study = optuna.create_study(direction='maximize') # Always maximize, negate drawdown in objective

        # Run the optimization
        try:
            study.optimize(optuna_objective_with_ranges, n_trials=n_trials)
            optuna_status_text.text("Optuna optimization finished.")
            optuna_progress_bar.progress(1.0) # Ensure progress bar is full
            st.success("Optimization finished.")
            st.toast("Optuna optimization finished!", icon="✅")
            # print("Debug: Optuna optimization finished successfully.")

            # Store Optuna results in session state for saving
            st.session_state['last_analysis_results'] = {
                'mode': 'Optuna',
                'study': study,
                'optimization_metric': analysis_metric
            }

        except optuna.exceptions.TrialPruned:
            optuna_status_text.text("Optuna optimization stopped by user.")
            st.warning("Optimization stopped by user.")
            st.toast("Optimization stopped by user.", icon="✋")
            # print("Debug: Optuna optimization stopped by user.")
            # Store partial results if stopped? Optuna study object should contain completed trials.
            st.session_state['last_analysis_results'] = {
                'mode': 'Optuna',
                'study': study, # Study object contains completed trials
                'optimization_metric': analysis_metric,
                'stopped': True
            }

        except Exception as e:
                optuna_status_text.text("Optuna optimization failed.")
                st.error(f"An error occurred during Optuna optimization: {e}")
                st.toast(f"Optuna optimization failed: {e}", icon="❌")
                # print(f"Debug: Error during Optuna optimization: {e}")
                # Clear results if analysis fails completely
                st.session_state['last_analysis_results'] = None
                st.stop()
        finally:
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag


        # --- Display Optimization Results ---
        st.subheader("Optimization Results")

        if st.session_state['last_analysis_results'] and st.session_state['last_analysis_results']['mode'] == 'Optuna':
            study_to_display = st.session_state['last_analysis_results']['study']
            metric_to_display = st.session_state['last_analysis_results']['optimization_metric']

            if study_to_display.trials: # Only display results if there were trials run
                st.write("Best Trial:")
                st.write(f"  Value: {study_to_display.best_value:.4f}")
                st.write("  Parameters:")
                st.write(study_to_display.best_params)

                # Display all trials in a DataFrame
                trials_df = study_to_display.trials_dataframe()

                # Rename the 'value' column to the optimization metric name
                trials_df.rename(columns={'value': metric_to_display}, inplace=True)

                # Format metrics for display
                # Need to convert negative drawdown back to positive for display
                if metric_to_display == "Max Drawdown (%)":
                        if metric_to_display in trials_df.columns:
                            trials_df[metric_to_display] = trials_df[metric_to_display].abs()

                # Apply formatting to relevant columns
                for col in ["Total Return", "Annualized Return", "Max Drawdown (%)"]:
                        # Check if the column exists before formatting
                        if col in trials_df.columns:
                            # Handle potential None/NaN values before formatting
                            trials_df[col] = trials_df[col].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')

                for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
                        if col in trials_df.columns:
                            trials_df[col] = trials_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')


                st.subheader("All Trials")
                st.dataframe(trials_df, key='optuna_trials_table')


                # --- Optuna Visualizations ---
                st.subheader("Optimization Visualizations")

                try:
                    # Plot Optimization History
                    fig_history = optuna.visualization.plot_optimization_history(study_to_display)
                    st.plotly_chart(fig_history)
                except Exception as e:
                    st.warning(f"Could not generate optimization history plot: {e}")
                    # print(f"Debug: Error generating optimization history plot: {e}")


                try:
                    # Plot Parameter Relationships (e.g., Parallel Coordinate Plot)
                    # Choose a subset of parameters if there are many
                    # Need to get the parameter names from the study's trials or the original config
                    param_names_for_plot = list(parameter_ranges_for_optuna.keys()) # Use the keys from the ranges used for optimization
                    if param_names_for_plot and len(param_names_for_plot) <= 10: # Limit to a reasonable number of parameters for the plot
                            fig_parallel = optuna.visualization.plot_parallel_coordinate(study_to_display, params=param_names_for_plot)
                            st.plotly_chart(fig_parallel)
                    else:
                        st.info(f"Skipping parallel coordinate plot as there are too many parameters ({len(param_names_for_plot)}).")
                        # print(f"Debug: Skipping parallel coordinate plot due to too many parameters ({len(param_names_for_plot)}).")


                except Exception as e:
                    st.warning(f"Could not generate parameter relationships plot: {e}")
                    # print(f"Debug: Error generating parameter relationships plot: {e}")


                try:
                    # Plot Slice Plot for individual parameter impact
                    param_names_for_plot = list(parameter_ranges_for_optuna.keys()) # Use the keys from the ranges used for optimization
                    if param_names_for_plot: # Only plot if there are parameters
                        fig_slice = optuna.visualization.plot_slice(study_to_display, params=param_names_for_plot)
                        st.plotly_chart(fig_slice)
                    else:
                        st.info("Skipping slice plot as there are no parameters to optimize.")
                        # print("Debug: Skipping slice plot as there are no parameters to optimize.")

                except Exception as e:
                    st.warning(f"Could not generate slice plot: {e}")
                    # print(f"Debug: Error generating slice plot: {e}")

                # Add other Optuna plots as desired (e.g., plot_contour, plot_essential_slice)
            else:
                st.warning("No trials were completed during optimization.")
        else:
                st.info("No Optuna optimization results available to display.")


    elif analysis_mode == "Walk-Forward Testing":
        st.info("Running Walk-Forward Testing...")
        st.toast("Running Walk-Forward Testing...", icon="⚙️")
        # print("Debug: Running Walk-Forward Testing.")

        # Check if there's enough data for at least one window
        total_days = len(list(data.values())[0]) if data and list(data.values())[0] is not None else 0
        if total_days < training_window_size + testing_window_size:
                st.error("Not enough data for at least one training and testing window with the specified sizes.")
                st.toast("Not enough data for Walk-Forward windows.", icon="❌")
                # print("Debug: Not enough data for Walk-Forward windows.")
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                st.stop()

        walk_forward_results = []
        current_start_index = 0
        window_count = 0

        # Calculate total number of windows for progress bar
        total_windows = 0
        temp_index = 0
        while temp_index + training_window_size + testing_window_size <= total_days:
                total_windows += 1
                temp_index += step_size

        if total_windows == 0:
                st.warning("No complete Walk-Forward windows can be created with the current settings and data.")
                st.toast("No complete Walk-Forward windows can be created.", icon="⚠️")
                # print("Debug: No complete Walk-Forward windows.")
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                st.stop()


        wf_progress_bar = st.progress(0)
        wf_status_text = st.empty()


        # Define the objective function for the internal Optuna optimization in each window
        def wf_optuna_objective(trial):
                # Check stop flag before running internal Optuna trials
                if st.session_state.stop_walkforward:
                    raise optuna.exceptions.TrialPruned("Walk-Forward optimization stopped by user.")

                # Get the data for the current training window
                train_data_window = {symbol: df.iloc[current_start_index : current_start_index + training_window_size].copy() for symbol, df in data.items()}

                # Use the parameter ranges defined in the UI for the internal optimization
                # Filter parameter_config_ui for parameters with ranges
                parameter_ranges_for_wf_optuna = {
                    param_name: config for param_name, config in parameter_config_ui.items()
                    if all(key in config for key in ['start', 'end', 'step', 'type'])
                }
                if not parameter_ranges_for_wf_optuna:
                    st.warning("No valid parameter ranges configured for Walk-Forward Optimization window.")
                    st.toast("No valid parameter ranges for Walk-Forward Optuna window.", icon="❌")
                    # print("Debug: No valid parameter ranges for Walk-Forward Optuna window.")
                    return -np.inf # Return a poor value if no params to optimize

                return optuna_objective(trial, train_data_window, StrategyClass, initial_capital, analysis_stop_loss_pct,
                                analysis_take_profit_pct, analysis_position_sizing_method,
                                analysis_position_sizing_value, analysis_metric, parameter_ranges_for_wf_optuna)


        # Iterate through walk-forward windows
        while current_start_index + training_window_size + testing_window_size <= total_days and not st.session_state.stop_walkforward:
            window_count += 1
            wf_status_text.text(f"Processing Walk-Forward Window {window_count}/{total_windows}...")
            # print(f"Debug: Processing Walk-Forward Window {window_count}/{total_windows}...")

            # --- Training Phase (Optimize on training window) ---
            train_start_date = list(data.values())[0].index[current_start_index]
            train_end_date = list(data.values())[0].index[current_start_index + training_window_size - 1]
            st.write(f"  Window {window_count} Training: {train_start_date.date()} to {train_end_date.date()}")

            # Create and run Optuna study for the training window
            study_train = optuna.create_study(direction='maximize')
            try:
                study_train.optimize(wf_optuna_objective, n_trials=wf_optuna_trials)
                best_params_train = study_train.best_params
                st.write(f"  Best params found in training: {best_params_train}")
                # print(f"Debug: Window {window_count} Training finished. Best params: {best_params_train}")

            except optuna.exceptions.TrialPruned:
                    st.warning(f"Walk-Forward Optimization stopped by user in training window {window_count}.")
                    st.toast(f"Walk-Forward Optimization stopped in window {window_count}.", icon="✋")
                    # print(f"Debug: Walk-Forward Optimization stopped by user in training window {window_count}.")
                    break # Break the while loop if stopped
            except Exception as e:
                    st.error(f"Error during Optuna optimization in training window {window_count}: {e}")
                    st.toast(f"Error in training window {window_count}: {e}", icon="❌")
                    # print(f"Debug: Error during Walk-Forward training window {window_count} optimization: {e}")
                    # Skip testing phase for this window if training fails
                    current_start_index += step_size
                    progress_value = window_count / total_windows
                    wf_progress_bar.progress(min(progress_value, 1.0))
                    time.sleep(0.05) # Add a small sleep
                    continue # Move to the next window


            # --- Testing Phase (Test best params on testing window) ---
            test_start_index = current_start_index + training_window_size
            test_end_index = test_start_index + testing_window_size - 1
            test_start_date = list(data.values())[0].index[test_start_index]
            test_end_date = list(data.values())[0].index[test_start_index + testing_window_size - 1] # Corrected end date calculation
            st.write(f"  Window {window_count} Testing: {test_start_date.date()} to {test_end_date.date()}")
            # print(f"Debug: Window {window_count} Testing: {test_start_date.date()} to {test_end_date.date()}")

            test_data_window = {symbol: df.iloc[test_start_index : test_end_index + 1].copy() for symbol, df in data.items()}

            # Instantiate strategy with best params from training
            try:
                strategy_instance_test = StrategyClass(**best_params_train)
                # Run backtest on the testing window
                backtester = Backtester(initial_capital=initial_capital) # Instantiate backtester for each window
                results_test = backtester.run_backtest(
                    test_data_window,
                    strategy_instance_test,
                    stop_loss_pct=analysis_stop_loss_pct,
                    take_profit_pct=analysis_take_profit_pct,
                    position_sizing_method=analysis_position_sizing_method,
                    position_sizing_value=analysis_position_sizing_value
                )

                if "error" in results_test:
                    st.warning(f"  Backtest failed in testing window {window_count}: {results_test['error']}")
                    st.toast(f"Backtest failed in testing window {window_count}.", icon="❌")
                    # print(f"Debug: Backtest failed in testing window {window_count}: {results_test['error']}")
                    window_metric_value = None
                    window_results = {
                            "Window": window_count,
                            "Training Period": f"{train_start_date.date()} to {train_end_date.date()}",
                            "Testing Period": f"{test_start_date.date()} to {test_end_date.date()}",
                            "Best Params (Training)": best_params_train,
                            "Error": results_test['error']
                    }
                else:
                    # Get the value of the analysis metric for the testing window
                    metric_key = analysis_metric.replace(' (%)', '').replace(' ', '_').lower()
                    window_metric_value = results_test.get(metric_key)
                    # print(f"Debug: Window {window_count} Testing finished. Metric ({analysis_metric}): {window_metric_value}")


                    window_results = {
                        "Window": window_count,
                        "Training Period": f"{train_start_date.date()} to {train_end_date.date()}",
                        "Testing Period": f"{test_start_date.date()} to {test_end_date.date()}",
                        "Best Params (Training)": best_params_train,
                        analysis_metric: window_metric_value, # Metric on testing window
                        "Total Return (Test)": results_test.get('total_return'),
                        "Annualized Return (Test)": results_test.get('annualized_return'),
                        "Sharpe Ratio (Test)": results_test.get('sharpe_ratio'),
                        "Sortino Ratio (Test)": results_test.get('sortino_ratio'),
                        "Calmar Ratio (Test)": results_test.get('calmar_ratio'),
                        "Max Drawdown (%) (Test)": results_test.get('max_drawdown_pct'),
                        "Trade Count (Test)": results_test.get('trade_count')
                    }
                walk_forward_results.append(window_results)


            except Exception as e:
                    st.error(f"Error running backtest in testing window {window_count}: {e}")
                    st.toast(f"Error in testing window {window_count}: {e}", icon="❌")
                    # print(f"Debug: Error running backtest in testing window {window_count}: {e}")
                    walk_forward_results.append({
                    "Window": window_count,
                    "Training Period": f"{train_start_date.date()} to {train_end_date.date()}",
                    "Testing Period": f"{test_start_date.date()} to {test_end_date.date()}",
                    "Best Params (Training)": best_params_train,
                    "Error": str(e) # Indicate failure
                })


            # Move to the next window
            current_start_index += step_size
            # Update progress bar after processing each window
            progress_value = window_count / total_windows
            wf_progress_bar.progress(min(progress_value, 1.0))
            time.sleep(0.05) # Add a small sleep


        wf_status_text.text("Walk-Forward Testing finished.")
        wf_progress_bar.progress(1.0) # Ensure progress bar is full
        st.success("Walk-Forward Testing finished!")
        st.toast("Walk-Forward Testing finished!", icon="✅")
        # print("Debug: Walk-Forward Testing finished.")

        # Store Walk-Forward results in session state for saving
        st.session_state['last_analysis_results'] = {
            'mode': 'Walk-Forward',
            'results': walk_forward_results,
            'optimization_metric': analysis_metric # Store metric used for optimization
        }

        st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag


        # --- Display Walk-Forward Results ---
        if st.session_state['last_analysis_results'] and st.session_state['last_analysis_results']['mode'] == 'Walk-Forward':
            results_to_display = st.session_state['last_analysis_results']['results']
            metric_used = st.session_state['last_analysis_results']['optimization_metric']

            if results_to_display:
                st.subheader("Walk-Forward Testing Results")
                results_df = pd.DataFrame(results_to_display)

                # Format metrics
                for col in ["Total Return (Test)", "Annualized Return (Test)", "Max Drawdown (%) (Test)"]:
                    if col in results_df.columns:
                        results_df[col] = results_df[col].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')
                for col in ["Sharpe Ratio (Test)", "Sortino Ratio (Test)", "Calmar Ratio (Test)"]:
                        if col in results_df.columns:
                            results_df[col] = results_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

                # Format Parameters column
                if 'Best Params (Training)' in results_df.columns:
                        results_df['Best Params (Training)'] = results_df['Best Params (Training)'].apply(lambda x: str(x))

                # Format Trade Count (Test)
                if 'Trade Count (Test)' in results_df.columns:
                        results_df['Trade Count (Test)'] = results_df['Trade Count (Test)'].apply(lambda x: f'{int(x):,}' if pd.notna(x) and pd.api.types.is_numeric_dtype(type(x)) else 'N/A')


                st.dataframe(results_df, key='wf_results_table')

                # Optional: Aggregate results (e.g., average metric across windows)
                st.subheader("Aggregated Walk-Forward Metrics (Average on Test Windows)")
                # Need to convert formatted columns back to numeric or store raw metric values separately
                # Or calculate averages from the raw values in walk_forward_results list
                # For simplicity, let's calculate averages from the raw values in results_to_display
                aggregated_metrics = {}
                for metric_name in ["Total Return", "Annualized Return", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown (%)", "Trade Count"]:
                        # Use the key as it appears in the results_to_display list of dicts
                        key_name = f"{metric_name} (Test)" if metric_name != "Trade Count" else "Trade Count (Test)"
                        raw_values = [r.get(key_name) for r in results_to_display if r.get(key_name) is not None and isinstance(r.get(key_name), (int, float))]
                        if raw_values:
                            aggregated_metrics[metric_name] = np.mean(raw_values)
                        else:
                            aggregated_metrics[metric_name] = "N/A"

                aggregated_df = pd.DataFrame([aggregated_metrics]).T.rename(columns={0: "Average Value"})

                # Format aggregated metrics
                for idx in ["Total Return", "Annualized Return", "Max Drawdown (%)"]:
                        if idx in aggregated_df.index and isinstance(aggregated_df.loc[idx, "Average Value"], (int, float)):
                            aggregated_df.loc[idx, "Average Value"] = f'{aggregated_df.loc[idx, "Average Value"]:.2%}'
                for idx in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
                        if idx in aggregated_df.index and isinstance(aggregated_df.loc[idx, "Average Value"], (int, float)):
                            aggregated_df.loc[idx, "Average Value"] = f'{aggregated_df.loc[idx, "Average Value"]:.2f}'
                if "Trade Count" in aggregated_df.index and isinstance(aggregated_df.loc["Trade Count", "Average Value"], (int, float)):
                        aggregated_df.loc["Trade Count", "Average Value"] = f'{int(aggregated_df.loc["Trade Count", "Average Value"]):,}'


                st.dataframe(aggregated_df, key='wf_aggregated_metrics')


            else:
                st.warning("No successful walk-forward windows were processed.")
                # print("Debug: No successful walk-forward windows processed.")
        else:
                st.info("No Walk-Forward Testing results available to display.")


    elif analysis_mode == "Parameter Sensitivity Analysis":
        st.info("Running Parameter Sensitivity Analysis...")
        st.toast("Running Parameter Sensitivity Analysis...", icon="⚙️")
        # print("Debug: Running Parameter Sensitivity Analysis.")

        sensitivity_params = st.session_state.get('sensitivity_params', [])
        if not sensitivity_params:
                st.warning("No parameters selected for sensitivity analysis.")
                st.toast("No parameters selected for sensitivity analysis.", icon="⚠️")
                # print("Debug: No sensitivity parameters selected.")
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                st.stop()


        sensitivity_results = []
        total_combinations = 0 # Will calculate below

        # Generate parameter combinations for sensitivity analysis
        # Parameters selected for sensitivity will use their defined ranges
        # Parameters NOT selected will use their 'start' value as a fixed value
        sensitivity_param_values = {}
        fixed_params = {}

        for param_name, param_config in parameter_config_ui.items():
                if param_name in sensitivity_params:
                    # Create the range of values for this sensitivity parameter
                    start_value = param_config.get('start') # Use .get for safety
                    end_value = param_config.get('end') # Use .get for safety
                    step_value = param_config.get('step') # Use .get for safety
                    param_type = param_config.get('type') # Use .get for safety

                    if start_value is None or end_value is None or step_value is None or param_type is None:
                        st.warning(f"Sensitivity range not fully defined for parameter '{param_name}'. Skipping.")
                        st.toast(f"Sensitivity range not fully defined for '{param_name}'.", icon="⚠️")
                        # print(f"Debug: Sensitivity range not fully defined for '{param_name}'.")
                        continue # Skip this parameter if range is incomplete


                    if param_type == int:
                        # Ensure step is not zero
                        if step_value == 0:
                                st.warning(f"Step for sensitivity parameter '{param_name}' cannot be zero. Using a single value.")
                                st.toast(f"Step for '{param_name}' cannot be zero.", icon="⚠️")
                                # print(f"Debug: Step is zero for '{param_name}'.")
                                sensitivity_param_values[param_name] = [start_value]
                        else:
                                # Add 1 to end_value to include the end value in the range
                                sensitivity_param_values[param_name] = list(range(int(start_value), int(end_value) + int(step_value), int(step_value)))
                    elif param_type == float:
                        # Ensure step is not zero
                        if step_value == 0.0:
                                st.warning(f"Step for sensitivity parameter '{param_name}' cannot be zero. Using a single value.")
                                st.toast(f"Step for '{param_name}' cannot be zero.", icon="⚠️")
                                # print(f"Debug: Step is zero for float '{param_name}'.")
                                sensitivity_param_values[param_name] = [start_value]
                        else:
                                # Calculate number of steps and generate values
                                num_steps = int(round((end_value - start_value) / step_value))
                                sensitivity_param_values[param_name] = [start_value + i * step_value for i in range(num_steps + 1)]
                                # Ensure the end value is included due to potential floating point inaccuracies
                                if abs(sensitivity_param_values[param_name][-1] - end_value) > 1e-9:
                                    sensitivity_param_values[param_name].append(end_value)

                                sensitivity_param_values[param_name] = [round(val, 4) for val in sensitivity_param_values[param_name]] # Round to avoid floating point display issues

                    else:
                        # Non-numeric sensitivity parameter - treat as a single value
                        st.warning(f"Sensitivity range not applicable for parameter type of '{param_name}'. Will use single value: {start_value}")
                        st.toast(f"Sensitivity range not applicable for '{param_name}'.", icon="⚠️")
                        # print(f"Debug: Non-numeric type for sensitivity parameter '{param_name}'.")
                        sensitivity_param_values[param_name] = [start_value]

                    # Filter out duplicates and sort
                    if param_name in sensitivity_param_values: # Check if the parameter was successfully added
                        sensitivity_param_values[param_name] = sorted(list(set(sensitivity_param_values[param_name])))

                else:
                    # Fixed parameter for sensitivity analysis - use the 'start' value as fixed
                    if 'start' in param_config:
                        fixed_params[param_name] = param_config['start']
                    else:
                        # This case should be caught by the disabled check, but as a safeguard
                        st.error(f"Fixed value (Start) not defined for non-sensitivity parameter '{param_name}'. Cannot proceed.")
                        st.toast(f"Fixed value not defined for '{param_name}'.", icon="❌")
                        # print(f"Debug: Fixed value (Start) not defined for non-sensitivity parameter '{param_name}'.")
                        st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                        st.stop()


        # Calculate total combinations
        if sensitivity_param_values:
                total_combinations = np.prod([len(values) for values in sensitivity_param_values.values()])
        else:
                total_combinations = 1 # If no sensitivity parameters with valid ranges, run one backtest with fixed params

        if total_combinations == 0:
                st.warning("No valid parameter combinations to test for sensitivity analysis.")
                st.toast("No valid parameter combinations to test.", icon="⚠️")
                # print("Debug: No valid parameter combinations for sensitivity analysis.")
                st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag
                st.stop()


        st.write(f"Testing {total_combinations} parameter combinations.")
        # print(f"Debug: Testing {total_combinations} parameter combinations.")


        sens_progress_bar = st.progress(0)
        sens_status_text = st.empty()

        start_time = time.time()
        i = 0 # Counter for combinations

        # Generate combinations of sensitivity parameters
        sensitivity_param_names = list(sensitivity_param_values.keys())
        sensitivity_param_combos = list(itertools.product(*sensitivity_param_values.values()))


        for combo in sensitivity_param_combos:
                # Check stop flag before running each combination
                if st.session_state.stop_sensitivity:
                    sens_status_text.text("Sensitivity analysis stopped by user.")
                    st.warning("Sensitivity analysis stopped by user.")
                    st.toast("Sensitivity analysis stopped by user.", icon="✋")
                    # print("Debug: Sensitivity analysis stopped by user.")
                    break # Break the for loop


                current_params = fixed_params.copy() # Start with fixed parameters
                current_params.update(dict(zip(sensitivity_param_names, combo))) # Add sensitivity parameters

                sens_status_text.text(f"Running backtest {i + 1}/{total_combinations} with params: {current_params}")
                # print(f"Debug: Running backtest {i + 1}/{total_combinations} with params: {current_params}")


                try:
                    # Instantiate the strategy with the current parameter combination
                    strategy_instance = StrategyClass(**current_params)

                    # Run the multi-asset backtest
                    backtester = Backtester(initial_capital=initial_capital)
                    results = backtester.run_backtest(
                        data.copy(),
                        strategy_instance,
                        stop_loss_pct=analysis_stop_loss_pct,
                        take_profit_pct=analysis_take_profit_pct,
                        position_sizing_method=analysis_position_sizing_method,
                        position_sizing_value=analysis_position_sizing_value
                    )

                    if "error" in results:
                        st.warning(f"Backtest failed for params {current_params}: {results['error']}")
                        st.toast(f"Backtest failed for params {current_params}.", icon="❌")
                        # print(f"Debug: Backtest failed for params {current_params}: {results['error']}")
                        # Append results with None for metrics
                        sensitivity_results.append({
                            "Parameters": current_params,
                            "Total Return": None,
                            "Annualized Return": None,
                            "Sharpe Ratio": None,
                            "Sortino Ratio": None,
                            "Calmar Ratio": None,
                            "Max Drawdown (%)": None,
                            "Trade Count": None,
                            "Error": results['error'] # Include error message
                        })
                    else:
                        # Append results with calculated metrics
                        sensitivity_results.append({
                            "Parameters": current_params,
                            "Total Return": results.get('total_return'),
                            "Annualized Return": results.get('annualized_return'), # Corrected key
                            "Sharpe Ratio": results.get('sharpe_ratio'),
                            "Sortino Ratio": results.get('sortino_ratio'),
                            "Calmar Ratio": results.get('calmar_ratio'),
                            "Max Drawdown (%)": results.get('max_drawdown_pct'),
                            "Max Drawdown ($)": results.get('max_drawdown_value'), # Added drawdown value
                            "Trade Count": results.get('trade_count')
                        })
                        # print(f"Debug: Backtest successful for params {current_params}. Metrics: {sensitivity_results[-1]}")


                except Exception as e:
                    st.error(f"Error running backtest for params {current_params}: {e}")
                    st.toast(f"Error running backtest for params {current_params}: {e}", icon="❌")
                    # print(f"Debug: Error running backtest for params {current_params}: {e}")
                    # Append results with None for metrics
                    sensitivity_results.append({
                        "Parameters": current_params,
                        "Total Return": None,
                        "Annualized Return": None,
                        "Sharpe Ratio": None,
                        "Sortino Ratio": None,
                        "Calmar Ratio": None,
                        "Max Drawdown (%)": None,
                        "Trade Count": None,
                        "Error": str(e) # Include error message
                    })

                i += 1 # Increment counter
                sens_progress_bar.progress(i / total_combinations)
                time.sleep(0.05) # Add a small sleep

        # Ensure progress bar is full even if stopped
        sens_progress_bar.progress(i / total_combinations if total_combinations > 0 else 1.0)
        end_time = time.time()
        if not st.session_state.stop_sensitivity: # Only show finished message if not stopped by user
                sens_status_text.text(f"Parameter Sensitivity Analysis finished in {end_time - start_time:.2f} seconds.")
                st.success("Parameter Sensitivity Analysis finished!")
                st.toast("Parameter Sensitivity Analysis finished!", icon="✅")
                # print(f"Debug: Parameter Sensitivity Analysis finished in {end_time - start_time:.2f} seconds.")

        # Store Sensitivity results in session state for saving
        st.session_state['last_analysis_results'] = {
            'mode': 'Sensitivity',
            'results': sensitivity_results,
            'sensitive_parameters': sensitivity_params, # Store the names of sensitive parameters
            'analysis_metric': analysis_metric # Store the metric used for analysis
        }

        st.session_state[f'is_{analysis_mode.replace(" ", "_").lower()}_running'] = False # Reset running flag


        # --- Display Sensitivity Results ---
        if st.session_state['last_analysis_results'] and st.session_state['last_analysis_results']['mode'] == 'Sensitivity':
            results_to_display = st.session_state['last_analysis_results']['results']
            sensitive_params_displayed = st.session_state['last_analysis_results']['sensitive_parameters']
            metric_used = st.session_state['last_analysis_results']['analysis_metric']


            if results_to_display:
                st.subheader("Parameter Sensitivity Analysis Results")
                results_df = pd.DataFrame(results_to_display)

                # Format metrics for display
                for col in ["Total Return", "Annualized Return", "Max Drawdown (%)"]:
                        if col in results_df.columns:
                            results_df[col] = results_df[col].apply(lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A')

                for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
                        if col in results_df.columns:
                            results_df[col] = results_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

                # Format max_drawdown_value and trade_count as integers if they exist
                for col in ["Max Drawdown ($)", "Trade Count"]:
                        if col in results_df.columns:
                            results_df[col] = results_df[col].apply(lambda x: f'{int(x):,}' if pd.notna(x) and pd.api.types.is_numeric_dtype(type(x)) else 'N/A')


                # Handle the 'Parameters' column display (convert dict to string)
                if 'Parameters' in results_df.columns:
                        results_df['Parameters'] = results_df['Parameters'].apply(lambda x: str(x))

                st.subheader("Results Table")
                st.dataframe(results_df, key='sensitivity_results_table')

                # --- Sensitivity Visualizations ---
                st.subheader("Sensitivity Visualizations")

                # Plotting depends on the number of sensitive parameters
                sensitive_params_for_plotting = [p for p in sensitive_params_displayed if p in parameter_config_ui and parameter_config_ui[p].get('type') in [int, float]]


                if len(sensitive_params_for_plotting) == 1:
                        # 1D plot: Metric vs Parameter Value
                        param_name = sensitive_params_for_plotting[0]
                        metric_col_name = metric_used # Use the selected metric name

                        # Find the corresponding raw metric key
                        raw_metric_key = metric_used.replace(' (%)', '').replace(' ', '_').lower()

                        # Filter results to include only rows with valid numeric metric value and parameter value
                        plot_data = [(r['Parameters'].get(param_name), r.get(raw_metric_key)) for r in results_to_display
                                    if r.get(raw_metric_key) is not None and isinstance(r.get(raw_metric_key), (int, float))
                                    and param_name in r['Parameters'] and r['Parameters'].get(param_name) is not None
                                    and isinstance(r['Parameters'].get(param_name), (int, float))] # Ensure parameter is also numeric


                        if len(plot_data) >= 2: # Need at least two points for a line plot
                            param_values = [item[0] for item in plot_data]
                            raw_metric_values = [item[1] for item in plot_data]

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=param_values, y=raw_metric_values, mode='lines+markers', name=metric_col_name))
                            fig.update_layout(
                                title=f'{metric_col_name} vs {param_name}',
                                xaxis_title=param_name,
                                yaxis_title=metric_col_name
                            )
                            st.plotly_chart(fig)
                            # print(f"Debug: Generated 1D sensitivity plot for {param_name}.")
                        else:
                            st.warning(f"Could not generate 1D sensitivity plot for {param_name} and {metric_used}. Ensure valid numeric data and at least two data points.")
                            # print(f"Debug: Could not generate 1D sensitivity plot for {param_name}.")


                elif len(sensitive_params_for_plotting) == 2:
                        # 2D plot: Metric as a surface/heatmap over two parameters
                        param_name_x = sensitive_params_for_plotting[0]
                        param_name_y = sensitive_params_for_plotting[1]
                        metric_col_name = metric_used

                        # Need to reshape results into a grid for heatmap/surface plot
                        # This requires that the parameter combinations form a complete grid
                        try:
                            # Create a list of dictionaries suitable for DataFrame creation
                            # Include the raw metric value for plotting
                            raw_metric_key = metric_used.replace(' (%)', '').replace(' ', '_').lower()
                            plot_data_list = []
                            for r in results_to_display:
                                # Ensure both parameters and the metric are numeric and not None
                                if r.get(raw_metric_key) is not None and isinstance(r.get(raw_metric_key), (int, float)) \
                                    and param_name_x in r['Parameters'] and r['Parameters'].get(param_name_x) is not None and isinstance(r['Parameters'].get(param_name_x), (int, float)) \
                                    and param_name_y in r['Parameters'] and r['Parameters'].get(param_name_y) is not None and isinstance(r['Parameters'].get(param_name_y), (int, float)):
                                        plot_data_list.append({
                                            param_name_x: r['Parameters'].get(param_name_x),
                                            param_name_y: r['Parameters'].get(param_name_y),
                                            'MetricValue': r.get(raw_metric_key) # Use a generic name for the metric value column
                                        })

                            if plot_data_list:
                                plot_df = pd.DataFrame(plot_data_list)

                                # Create a pivot table for the metric
                                pivot_table = plot_df.pivot_table(
                                    values='MetricValue',
                                    index=param_name_y,
                                    columns=param_name_x
                                )

                                if not pivot_table.empty:
                                    fig = go.Figure(data=go.Heatmap(
                                        z=pivot_table.values,
                                        x=pivot_table.columns,
                                        y=pivot_table.index,
                                        colorscale='Viridis' # Or another colorscale
                                    ))
                                    fig.update_layout(
                                        title=f'{metric_col_name} Sensitivity to {param_name_x} and {param_name_y}',
                                        xaxis_title=param_name_x,
                                        yaxis_title=param_name_y
                                    )
                                    st.plotly_chart(fig)
                                    # print(f"Debug: Generated 2D sensitivity plot for {param_name_x} and {param_name_y}.")
                                else:
                                    st.warning(f"Could not generate 2D sensitivity plot for {param_name_x}, {param_name_y}, and {metric_used}. Ensure data forms a grid and metric values are valid.")
                                    # print(f"Debug: Could not generate 2D sensitivity plot for {param_name_x}, {param_name_y}.")


                            else:
                                st.warning(f"Could not generate 2D sensitivity plot for {param_name_x}, {param_name_y}, and {metric_used}. No valid data points found.")
                                # print(f"Debug: No valid data points for 2D sensitivity plot for {param_name_x}, {param_name_y}.")


                        except Exception as e:
                            st.warning(f"Could not generate 2D sensitivity plot: {e}. Ensure parameter combinations form a complete grid.")
                            # print(f"Debug: Error generating 2D sensitivity plot: {e}")


                else:
                        st.info("Sensitivity visualization is best with 1 or 2 parameters.")
                        # print("Debug: Skipping sensitivity visualization as parameters for plotting are not 1 or 2.")

            else:
                st.warning("No successful backtest results were obtained for sensitivity analysis.")
                # print("Debug: No successful backtest results for sensitivity analysis.")
        else:
                st.info("No Parameter Sensitivity Analysis results available to display.")


# --- Save Analysis Results Button(s) ---
# Display save buttons based on the analysis mode and if results exist in session state
if st.session_state['last_analysis_results'] is not None:
    analysis_results_to_save = st.session_state['last_analysis_results']
    analysis_mode_completed = analysis_results_to_save.get('mode')

    # Use the placeholder to render the download buttons
    with save_button_placeholder.container():
        st.markdown("---") # Add a separator before save buttons
        st.subheader("Download Analysis Results")

        if analysis_mode_completed == 'Optuna' and 'study' in analysis_results_to_save:
            study_to_save = analysis_results_to_save['study']
            if study_to_save.trials:
                # Save Optuna study trials as CSV
                trials_df = study_to_save.trials_dataframe()
                # Rename the 'value' column to the optimization metric name for clarity in the saved file
                metric_name_for_save = analysis_results_to_save.get('optimization_metric', 'Value').replace(' (%)', '').replace(' ', '_').lower()
                trials_df.rename(columns={'value': metric_name_for_save}, inplace=True)

                # Save best trial parameters as JSON
                best_params_json = json.dumps(study_to_save.best_params, indent=4)

                col_trials, col_best_params = st.columns(2)

                with col_trials:
                    st.download_button(
                        label="Optuna Trials (CSV)",
                        data=trials_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"optuna_trials_{datetime.date.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key='download_optuna_trials_csv',
                        on_click=lambda: st.toast("Optuna Trials CSV prepared for download!", icon="💾")
                    )
                with col_best_params:
                     st.download_button(
                        label="Best Params (JSON)",
                        data=best_params_json,
                        file_name=f"optuna_best_params_{datetime.date.today().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        key='download_optuna_best_params_json',
                        on_click=lambda: st.toast("Best Params JSON prepared for download!", icon="💾")
                    )

            else:
                 st.info("No Optuna trials completed to save.")


        elif analysis_mode_completed == 'Walk-Forward' and 'results' in analysis_results_to_save:
            wf_results_list = analysis_results_to_save['results']
            if wf_results_list:
                wf_results_df = pd.DataFrame(wf_results_list)
                # Convert 'Best Params (Training)' column to string for CSV compatibility
                if 'Best Params (Training)' in wf_results_df.columns:
                     wf_results_df['Best Params (Training)'] = wf_results_df['Best Params (Training)'].astype(str)

                st.download_button(
                    label="Walk-Forward Results (CSV)",
                    data=wf_results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"walk_forward_results_{datetime.date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key='download_walk_forward_results_csv',
                    on_click=lambda: st.toast("Walk-Forward Results CSV prepared for download!", icon="💾")
                )
            else:
                 st.info("No Walk-Forward results to save.")


        elif analysis_mode_completed == 'Sensitivity' and 'results' in analysis_results_to_save:
            sens_results_list = analysis_results_to_save['results']
            if sens_results_list:
                sens_results_df = pd.DataFrame(sens_results_list)
                # Convert 'Parameters' column (which is a dict) to string for CSV compatibility
                if 'Parameters' in sens_results_df.columns:
                     sens_results_df['Parameters'] = sens_results_df['Parameters'].astype(str)

                st.download_button(
                    label="Sensitivity Results (CSV)",
                    data=sens_results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"sensitivity_results_{datetime.date.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key='download_sensitivity_results_csv',
                    on_click=lambda: st.toast("Sensitivity Results CSV prepared for download!", icon="💾")
                )
            else:
                 st.info("No Sensitivity results to save.")

