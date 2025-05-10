import streamlit as st
import pandas as pd
import datetime
import sys
import os
import matplotlib.pyplot as plt # Keep import if plot_price_with_signals is used elsewhere
from streamlit_searchbox import st_searchbox
import plotly.graph_objects as go # Import Plotly for interactive charts
import json # Import json for saving results
import plotly.express as px # Import for potential pie chart


# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import necessary modules
from data_sources.base_data_source import BaseDataSource
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource # Import specific data source for searchbox
from strategies.base_strategy import BaseStrategy
from backtester.backtester import Backtester
from utils.plotting import plot_equity_curve # Import the modified plot_equity_curve


# --- Helper function for symbol searchbox calling the data source method ---
# Modified to accept **kwargs to handle unexpected arguments like 'multiple'
def search_symbols_portfolio_management(search_term: str, **kwargs) -> list[str]:
    """
    Calls the search_symbols method of the currently selected data source for the portfolio management page.
    Accepts arbitrary keyword arguments to handle unexpected inputs from the searchbox component.
    Accesses AVAILABLE_DATA_SOURCES and selected_data_source_name from Streamlit's session state.
    """
    # Optional: Print kwargs to debug what's being passed
    # print(f"search_symbols_portfolio_management called with search_term='{search_term}' and kwargs={kwargs}")

    AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
    if not AVAILABLE_DATA_SOURCES:
        print("Error: AVAILABLE_DATA_SOURCES not found in session state.")
        return ["Error: Data sources not loaded."]

    selected_data_source_name = st.session_state.get('selected_data_source_name', list(st.session_state.get('available_data_sources', {}).keys())[0])
    DataSourceClass = AVAILABLE_DATA_SOURCES.get(selected_data_source_name)

    if DataSourceClass and hasattr(DataSourceClass, 'search_symbols') and callable(getattr(DataSourceClass, 'search_symbols')):
        try:
            data_source_instance = DataSourceClass() # Instantiate the selected data source
            return data_source_instance.search_symbols(search_term)
        except Exception as e:
            print(f"Error calling search_symbols for {selected_data_source_name}: {e}")
            return [f"Error searching symbols: {e}"]
    else:
        # Fallback for data sources that don't implement search_symbols
        # If the data source has get_available_symbols, filter that list (less efficient for large lists)
        if DataSourceClass and hasattr(DataSourceClass, 'get_available_symbols') and callable(getattr(DataSourceClass, 'get_available_symbols')):
             try:
                  data_source_instance = DataSourceClass()
                  all_symbols = data_source_instance.get_available_symbols()
                  if not search_term:
                       return all_symbols[:20] # Return first 20 if search term is empty
                  search_term_lower = search_term.lower()
                  return [symbol for symbol in all_symbols if search_term_lower in symbol.lower()][:20]
             except Exception as e:
                  print(f"Error calling get_available_symbols for {selected_data_source_name}: {e}")
                  return [f"Error listing symbols: {e}"]
        else:
             return [f"Search not available for {selected_data_source_name}"]

# --- Session State Initialization ---
# Initialize selected_symbols list and initial_allocation dictionary in session state
if 'portfolio_selected_symbols' not in st.session_state:
    st.session_state['portfolio_selected_symbols'] = []
if 'portfolio_initial_allocation' not in st.session_state:
    st.session_state['portfolio_initial_allocation'] = {}


# --- Helper function to add a symbol ---
def add_symbol_to_portfolio(symbol):
    if symbol and symbol not in st.session_state['portfolio_selected_symbols']:
        st.session_state['portfolio_selected_symbols'].append(symbol)
        # Initialize allocation for the new symbol
        st.session_state['portfolio_initial_allocation'][symbol] = 0.0
        st.toast(f"Added {symbol} to portfolio.", icon="✅")
    elif symbol in st.session_state['portfolio_selected_symbols']:
        st.warning(f"{symbol} is already in the portfolio.")
        st.toast(f"{symbol} already added.", icon="⚠️")
    else:
        st.warning("Please select a symbol to add.")
        st.toast("Please select a symbol.", icon="⚠️")


# --- Helper function to remove a symbol ---
def remove_symbol_from_portfolio(symbol):
    if symbol in st.session_state['portfolio_selected_symbols']:
        st.session_state['portfolio_selected_symbols'].remove(symbol)
        # Remove allocation for the removed symbol
        if symbol in st.session_state['portfolio_initial_allocation']:
            del st.session_state['portfolio_initial_allocation'][symbol]
        st.toast(f"Removed {symbol} from portfolio.", icon="🗑️")


# --- UI Elements and Logic for the Portfolio Management and Backtest Page ---
# This code will be executed directly by Streamlit when the page is selected.

st.title("Portfolio Management and Backtest")

# Retrieve available strategies and data sources from session state
AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})
AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])


if not AVAILABLE_STRATEGIES:
    st.warning("No trading strategies available.")
    st.stop()


st.header("Portfolio Configuration and Backtest Execution")

st.markdown("""
Configure your portfolio assets and initial allocation, set backtest parameters, and run the simulation.
""")

# --- Use Expanders for better organization ---

# --- Data Selection and Capital Expander ---
with st.expander("Data Selection and Capital", expanded=True): # Start expanded
    st.subheader("Data Selection and Capital")
    selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])
    DataSourceClass = AVAILABLE_DATA_SOURCES.get(selected_data_source_name)
    if not DataSourceClass:
         st.error(f"Data source '{selected_data_source_name}' not found.")
         st.stop()
    # Instantiate the data source, passing API key if needed (handled during fetch)
    # data_source = DataSourceClass()


    st.write(f"Using Data Source: **{selected_data_source_name}** (Change in Settings)")


    col_date_start, col_date_end = st.columns(2)
    with col_date_start:
        today = datetime.date.today()
        start_date = st.date_input("Start Date", today - datetime.timedelta(days=365*2), key='portfolio_management_start_date')
    with col_date_end:
        end_date = st.date_input("End Date", today, key='portfolio_management_end_date')

    # Initial Capital - Moved here
    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=100000.0, step=1000.0, key='portfolio_management_initial_capital')


# --- Portfolio Assets and Allocation Expander ---
with st.expander("Portfolio Assets and Allocation", expanded=True): # Start expanded
    st.subheader("Portfolio Assets and Initial Allocation")
    st.write("Select symbols for your portfolio and specify their initial allocation percentage.")
    st.info("The remaining capital will be held as cash.")

    # --- Symbol Selection (Single searchbox + Add Button) ---
    col_search, col_add = st.columns([0.7, 0.3])
    with col_search:
        # Use a single select searchbox
        selected_symbol_to_add = st_searchbox(
            search_symbols_portfolio_management, # Use the dynamic search function
            label="Search and Select Symbol to Add",
            key="portfolio_management_symbol_searchbox_add",
            placeholder="Type to search for symbols...",
            # multiple=False is the default, no need to specify
        )
    with col_add:
        # Add button to add the selected symbol to the list
        st.markdown("<br>", unsafe_allow_html=True) # Add some space to align button
        st.button("Add Symbol", on_click=add_symbol_to_portfolio, args=(selected_symbol_to_add,), key='add_symbol_button')


    st.markdown("---") # Separator

    st.subheader("Current Portfolio Assets and Allocation (%)")

    # --- Display Selected Symbols and Allocation Inputs ---
    selected_symbols = st.session_state.get('portfolio_selected_symbols', [])
    initial_allocation = st.session_state.get('portfolio_initial_allocation', {})
    total_allocated_pct = 0.0

    if selected_symbols:
        # Use columns for better layout (Symbol | Allocation % | Remove Button)
        col_symbol, col_allocation, col_remove = st.columns([0.5, 0.25, 0.25]) # Adjust widths

        with col_symbol:
             st.markdown("**Symbol**")
        with col_allocation:
             st.markdown("**Allocation (%)**")
        with col_remove:
             st.markdown("**Remove**") # Header for remove button column


        # Iterate through the list of selected symbols to display allocation inputs and remove buttons
        for i, symbol in enumerate(selected_symbols):
            # Ensure the symbol is in the current allocation dictionary (handles adding new symbols)
            if symbol not in initial_allocation:
                 initial_allocation[symbol] = 0.0

            # Display symbol, allocation input, and remove button in columns
            with col_symbol:
                 st.write(symbol) # Display the symbol name

            with col_allocation:
                current_allocation = initial_allocation.get(symbol, 0.0)
                # Updated key to include index 'i' for uniqueness
                allocated_pct = st.number_input(
                    " ", # Use an empty label as the symbol is displayed in the other column
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_allocation),
                    step=0.1,
                    format="%.2f",
                    key=f'portfolio_management_allocation_{i}_{symbol}' # Unique key for this page
                )
                initial_allocation[symbol] = allocated_pct
                total_allocated_pct += allocated_pct

            with col_remove:
                 # Add a remove button for each symbol
                 st.button("Remove", on_click=remove_symbol_from_portfolio, args=(symbol,), key=f'remove_symbol_button_{symbol}')


        # Update session state with the latest allocation values after iterating through inputs
        # This is important because number_input updates session state on interaction
        st.session_state['portfolio_initial_allocation'] = initial_allocation


        st.markdown("---") # Separator after allocation inputs

        # Display total allocated percentage and remaining cash percentage
        st.markdown(f"**Total Allocated:** {total_allocated_pct:.2f}%")
        st.markdown(f"**Remaining Cash:** {100.0 - total_allocated_pct:.2f}%")

        # Add a warning if total allocation is not 100% (or close to it)
        if abs(total_allocated_pct - 100.0) > 1e-6: # Allow for small floating point differences
            st.warning("Total allocation percentage is not 100%. The remaining percentage will be held as cash.")
            st.toast("Total allocation is not 100%.", icon="⚠️")

        # Optional: Add a simple pie chart visualization of allocation
        if selected_symbols and total_allocated_pct > 0:
             allocation_data = {s: initial_allocation.get(s, 0.0) for s in selected_symbols if initial_allocation.get(s, 0.0) > 0}
             if 100.0 - total_allocated_pct > 1e-6: # Add cash if remaining
                  allocation_data['Cash'] = 100.0 - total_allocated_pct

             if allocation_data:
                  # Filter out assets with 0% allocation for the chart
                  chart_data = {asset: pct for asset, pct in allocation_data.items() if pct > 0}
                  if chart_data:
                       allocation_df = pd.DataFrame(list(chart_data.items()), columns=['Asset', 'Percentage'])
                       fig = px.pie(allocation_df, values='Percentage', names='Asset', title='Initial Portfolio Allocation Distribution')
                       st.plotly_chart(fig, use_container_width=True)
                  else:
                       st.info("No assets with non-zero allocation to display in the chart.")

    else:
        # If no symbols selected, ensure allocation is reset and display info
        st.session_state['portfolio_initial_allocation'] = {}
        # st.session_state['portfolio_allocation_symbols'] = [] # No longer needed as selected_symbols is the source of truth
        st.info("Select symbols above to add them to your portfolio.")


# --- Backtest Parameters Expander ---
with st.expander("Backtest Parameters", expanded=True): # Start expanded
    st.header("Backtest Parameters")

    # Use a container for Strategy and Parameters
    with st.container(border=True):
        st.subheader("Strategy Configuration")
        selected_strategy_name = st.selectbox("Select Strategy", list(AVAILABLE_STRATEGIES.keys()), key='portfolio_management_strategy') # Unique key
        StrategyClass = AVAILABLE_STRATEGIES.get(selected_strategy_name) # Use .get for safety


        # --- Display Strategy Parameters (Removed nested expander) ---
        st.subheader("Configure Strategy Parameters")
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
                                current_value = st.session_state.get(f'portfolio_management_param_{selected_strategy_name}_{param_name}', default_value) # Unique key

                                if isinstance(default_value, int):
                                    strategy_params[param_name] = st.number_input(param_name, min_value=1, value=int(current_value), step=1, key=f'portfolio_management_param_{selected_strategy_name}_{param_name}_int') # Unique key
                                elif isinstance(default_value, float):
                                     strategy_params[param_name] = st.number_input(param_name, value=float(current_value), step=0.1, format="%.2f", key=f'portfolio_management_param_{selected_strategy_name}_{param_name}_float') # Unique key
                                elif isinstance(default_value, bool):
                                     param_value = st.checkbox(param_name, value=bool(current_value), key=f'portfolio_management_param_{selected_strategy_name}_{param_name}_bool') # Unique key
                                     strategy_params[param_name] = param_value # Ensure boolean value is stored
                                else:
                                     strategy_params[param_name] = st.text_input(param_name, str(current_value), key=f'portfolio_management_param_{selected_strategy_name}_{param_name}_text') # Unique key
                                # Store parameter value in session state for persistence
                                st.session_state[f'portfolio_management_param_{selected_strategy_name}_{param_name}'] = strategy_params[param_name]

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
        else:
            strategy_instance = None # Explicitly set to None if StrategyClass not found or params have error


    # Use a container for Risk Management and Position Sizing
    with st.container(border=True):
        st.subheader("Risk and Position Management")
        col_sl, col_tp = st.columns(2)
        with col_sl:
            stop_loss_pct = st.number_input("Global Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.2f", key='portfolio_management_stop_loss_pct') / 100.0 # Unique key
        with col_tp:
            take_profit_pct = st.number_input("Global Take Profit (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key='portfolio_management_take_profit_pct') / 100.0 # Unique key

        st.subheader("Position Sizing")
        position_sizing_method = st.selectbox(
            "Method",
            ['Full Equity', 'Fixed Amount', 'Percentage of Equity'],
            key='portfolio_management_position_sizing_method' # Unique key
        )
        position_sizing_value = 0.0
        if position_sizing_method == 'Fixed Amount':
            position_sizing_value = st.number_input("Amount ($)", min_value=1.0, value=1000.0, step=100.0, key='portfolio_management_position_sizing_amount') # Unique key
        elif position_sizing_method == 'Percentage of Equity':
            position_sizing_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.2f", key='portfolio_management_position_sizing_percentage') / 100.0 # Unique key


# --- Run Backtest Button (Placed outside expanders) ---
# Disable the run button if no symbols or strategy is selected, or if initial capital is invalid,
# or if position sizing is invalid, or if the strategy instance could not be created.
is_disabled = not selected_symbols or strategy_instance is None or initial_capital <= 0 or (position_sizing_method != 'Full Equity' and position_sizing_value <= 0)

st.markdown("---") # Add a separator before the button

# Add a placeholder for the Save Results button (will be populated in the Results section)
save_button_placeholder = st.empty()


if st.button("Run Portfolio Backtest", key='run_portfolio_management_backtest_main', disabled=is_disabled): # Unique key
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
        st.toast("Error: Start date must be before end date.", icon="❌")
    elif not selected_symbols:
         st.warning("Please select at least one symbol to run the backtest.")
         st.toast("Please select at least one symbol to run the backtest.", icon="⚠️")
    elif initial_capital <= 0: # Added check for initial capital
         st.warning("Initial capital must be greater than 0.")
         st.toast("Initial capital must be greater than 0.", icon="⚠️")
    elif position_sizing_method != 'Full Equity' and position_sizing_value <= 0:
         st.warning(f"Please enter a valid {position_sizing_method} value.")
         st.toast(f"Please enter a valid {position_sizing_method} value.", icon="⚠️")
    elif strategy_instance is None: # Added check here as well
         st.warning("Strategy could not be initialized. Check parameters.")
         st.toast("Strategy could not be initialized. Check parameters.", icon="❌")
    else:
        # --- Backtest Execution (Wrapped in spinner) ---
        with st.spinner("Running backtest..."):
            # Fetch Data for all selected symbols
            st.info(f"Fetching data for {', '.join(selected_symbols)} from {start_date} to {end_date} using {selected_data_source_name}...")
            st.toast(f"Fetching data for {', '.join(selected_symbols)}...", icon="⏳")
            data = {}
            fetch_errors = []

            # Get the selected data source class
            DataSourceClass = AVAILABLE_DATA_SOURCES.get(selected_data_source_name)

            if DataSourceClass:
                try:
                    # Instantiate the data source, passing the API key if it's Alpha Vantage
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

                    # Ensure selected_symbols is a list for iteration
                    symbols_to_fetch = selected_symbols if isinstance(selected_symbols, list) else [selected_symbols]

                    for symbol in symbols_to_fetch:
                        try:
                            asset_data = data_source_instance.fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                            if not asset_data.empty:
                                data[symbol] = asset_data
                            else:
                                fetch_errors.append(f"No data fetched for {symbol}")
                        except Exception as e:
                            fetch_errors.append(f"Error fetching data for {symbol}: {e}")

                except ValueError as e:
                     st.error(f"Data source initialization error: {e}")
                     st.toast(f"Data source error: {e}", icon="❌")
                     st.stop()
                except Exception as e:
                     st.error(f"An unexpected error occurred during data source instantiation: {e}")
                     st.toast(f"Data source error: {e}", icon="❌")
                     st.stop()
            else:
                 st.error(f"Data source '{selected_data_source_name}' not found.")
                 st.toast(f"Data source '{selected_data_source_name}' not found.", icon="❌")
                 st.stop()


            if fetch_errors:
                for error in fetch_errors:
                    st.error(error)
                    st.toast(f"Error fetching data: {error}", icon="❌")
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
            # if data: st.dataframe(list(data.values())[0].head(), key='data_head') # Use a generic key


            # --- Run Backtest ---
            st.info(f"Running backtest with {strategy_instance.name} on {list(data.keys())}...")
            st.toast(f"Running backtest with {strategy_instance.name}...", icon="⚙️")
            backtester = Backtester(initial_capital=initial_capital)

            # --- Get Initial Allocation from Session State ---
            # The Backtester's internal logic currently does NOT use this
            # initial_allocation to set up initial positions. It assumes all capital
            # starts as cash. This is a placeholder for future enhancement
            # of the Backtester class itself to handle initial portfolio distribution.
            initial_allocation_for_backtester = st.session_state.get('portfolio_initial_allocation', {})


            try:
                results = backtester.run_backtest(
                    data.copy(),
                    strategy_instance,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    position_sizing_method=position_sizing_method,
                    position_sizing_value=position_sizing_value,
                    # Pass the initial allocation (currently not used by Backtester logic)
                    initial_allocation=initial_allocation_for_backtester
                )

                if "error" in results:
                    st.error(f"Backtest Error: {results['error']}")
                    st.toast(f"Backtest failed: {results['error']}", icon="❌")
                    # Clear previous results if backtest fails
                    if 'last_portfolio_management_backtest_results' in st.session_state: # Unique key for this page
                        del st.session_state['last_portfolio_management_backtest_results']
                    if 'last_portfolio_management_backtest_config' in st.session_state: # Unique key for this page
                         del st.session_state['last_portfolio_management_backtest_config']
                else:
                    st.success("Backtest completed.")
                    st.toast("Backtest completed successfully!", icon="✅")

                    # Store results in session state for saving
                    st.session_state['last_portfolio_management_backtest_results'] = results # Unique key for this page
                    # Store current configuration used for this backtest run for saving purposes
                    st.session_state['last_portfolio_management_backtest_config'] = { # Unique key for this page
                        "strategy_name": selected_strategy_name,
                        "strategy_parameters": strategy_params, # Use the actual params used
                        "data_source": selected_data_source_name,
                        "symbols": selected_symbols,
                        "start_date": start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime.date) else str(start_date),
                        "end_date": end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else str(end_date),
                        "initial_capital": initial_capital,
                        "stop_loss_pct": stop_loss_pct,
                        "take_profit_pct": take_profit_pct,
                        "position_sizing_method": position_sizing_method,
                        "position_sizing_value": position_sizing_value,
                        "initial_allocation": initial_allocation_for_backtester # Store the allocation used
                    }

            except Exception as e:
                st.error(f"An unexpected error occurred during backtest execution or results processing: {e}")
                st.toast(f"An unexpected error occurred: {e}", icon="❌")
                # Clear previous results if backtest fails
                if 'last_portfolio_management_backtest_results' in st.session_state:
                    del st.session_state['last_portfolio_management_backtest_results']
                if 'last_portfolio_management_backtest_config' in st.session_state:
                     del st.session_state['last_portfolio_management_backtest_config']


# --- Results Section (Displayed below the button if results are available) ---
# This block is outside the button click check but checks session state for results
if 'last_portfolio_management_backtest_results' in st.session_state and st.session_state.get('last_portfolio_management_backtest_results') is not None:
    results = st.session_state['last_portfolio_management_backtest_results']

    st.markdown("---") # Separator before results
    st.header("Backtest Results")

    st.subheader("Performance Metrics (Portfolio Level)")
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


    st.subheader("Equity Curve (Portfolio Level)")
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
                   # Wrap the single equity curve in a dictionary for the plotting function
                   equity_curves_for_plotting = {f"{st.session_state.get('last_portfolio_management_backtest_config', {}).get('strategy_name', 'Portfolio')}": portfolio_total} # Use strategy name from config
                   try:
                       # Use the updated plot_equity_curve function which is interactive
                       fig = plot_equity_curve(equity_curves_for_plotting, initial_capital=initial_capital) # Use initial_capital from config? Or current UI? Using current UI for consistency.
                       if fig: # Check if the plotting function returned a figure
                           st.plotly_chart(fig) # Use st.plotly_chart for interactive plots
                           # No need to close Plotly figures with plt.close()
                       else:
                           st.warning("Could not generate interactive equity curve plot.")
                   except Exception as e:
                       st.error(f"An error occurred while plotting equity curve: {e}")
             else:
                  st.warning("No valid equity curve data available for plotting.")
        else:
             st.info("Portfolio history not available for plotting.")


    # --- Display Detailed Portfolio History ---
    if 'portfolio_history' in results and results['portfolio_history'] is not None:
         with st.expander("View Detailed Portfolio History"):
              portfolio_history_df = pd.DataFrame(results['portfolio_history'])
              # Ensure index is DatetimeIndex and formatted for display
              if not isinstance(portfolio_history_df.index, pd.DatetimeIndex):
                   try:
                        portfolio_history_df.index = pd.to_datetime(portfolio_history_df.index).strftime('%Y-%m-%d %H:%M:%S')
                   except Exception as e:
                        st.warning(f"Could not format index for detailed portfolio history: {e}")
                        # Keep original index if formatting fails

              st.dataframe(portfolio_history_df, key='portfolio_management_detailed_history') # Unique key
    else:
         st.info("Detailed portfolio history not available.")


    if 'trades' in results and results['trades']:
        with st.expander("View Trade Log (All Assets)"):
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
                # Add 'symbol' column if it exists in the trades data
                # This is crucial for multi-asset trade logs
                if 'symbol' in trades_df.columns:
                    pass # Keep the column if it's already there
                else:
                    # If 'symbol' is not in the trades data, add a placeholder or warning
                    st.warning("Trade log data is missing 'symbol' column.")
                    trades_df['symbol'] = 'N/A' # Add a placeholder column


                st.dataframe(trades_df, key='portfolio_management_trade_log') # Unique key
            else:
                st.warning("Trade log data is not in the expected format.")
                st.json(results['trades']) # Display raw data for debugging
    else:
         st.info("No trades recorded for this backtest.")

    # --- Save Results Button ---
    # Display the save button only if there are backtest results AND config in session state and they are not None
    if 'last_portfolio_management_backtest_results' in st.session_state and st.session_state.get('last_portfolio_management_backtest_results') is not None \
       and 'last_portfolio_management_backtest_config' in st.session_state and st.session_state.get('last_portfolio_management_backtest_config') is not None:

        results_to_save = st.session_state['last_portfolio_management_backtest_results'] # Unique key
        config_used = st.session_state['last_portfolio_management_backtest_config'] # Unique key

        # Retrieve selected symbol and strategy name from the config stored in session state
        strategy_name_for_filename = config_used.get('strategy_name', 'UnknownStrategy')
        # Use a generic name or combine symbols for portfolio filename
        symbols_for_filename = "_".join(config_used.get('symbols', ['UnknownSymbols']))


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
                "Symbols": config_used.get('symbols'),
                "Start Date": config_used.get('start_date'),
                "End Date": config_used.get('end_date'),
                "Initial Capital": config_used.get('initial_capital'),
                "Stop Loss (%)": config_used.get('stop_loss_pct') * 100 if config_used.get('stop_loss_pct') is not None else 'N/A',
                "Take Profit (%)": config_used.get('take_profit_pct') * 100 if config_used.get('take_profit_pct') is not None else 'N/A',
                "Position Sizing Method": config_used.get('position_sizing_method'),
                "Position Sizing Value": config_used.get('position_sizing_value') if config_used.get('position_sizing_method') != 'Full Equity' else 'N/A',
                "Initial Allocation": config_used.get('initial_allocation', 'Not specified') # Include initial allocation in config
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

        # Detailed Portfolio History DataFrame
        detailed_history_df = pd.DataFrame()
        if 'portfolio_history' in results_to_save and results_to_save['portfolio_history'] is not None:
            try:
                detailed_history_df = pd.DataFrame(results_to_save['portfolio_history'])
                # Ensure index is DatetimeIndex and formatted for saving
                if not isinstance(detailed_history_df.index, pd.DatetimeIndex):
                     detailed_history_df.index = pd.to_datetime(detailed_history_df.index)
                detailed_history_df.index = detailed_history_df.index.strftime('%Y-%m-%d %H:%M:%S')
                detailed_history_df.reset_index(inplace=True)
                detailed_history_df.rename(columns={'index': 'Timestamp'}, inplace=True)

            except Exception as e:
                st.warning(f"Could not process detailed portfolio history for saving: {e}")
                detailed_history_df = pd.DataFrame() # Empty DataFrame on error


        trades_df = pd.DataFrame()
        if 'trades' in results_to_save and results_to_save['trades'] is not None:
            # Assuming trades is a list of dictionaries or similar structure
            if isinstance(results_to_save['trades'], list) and all(isinstance(trade, dict) for trade in results_to_save['trades']):
                try:
                    trades_df = pd.DataFrame(results_to_save['trades'])
                    # Format timestamp columns to datetime if they exist
                    for col in ['entry_time', 'exit_time']: # Adjust column names as per your trades structure
                         if col in trades_df.columns:
                              # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
                              trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Add 'symbol' column if it exists in the trades data
                    # This is crucial for multi-asset trade logs
                    if 'symbol' in trades_df.columns:
                         pass # Keep the column if it's already there
                    else:
                         # If 'symbol' is not in the trades data, add a placeholder or warning
                         st.warning("Trade log data is missing 'symbol' column.")
                         trades_df['symbol'] = 'N/A' # Add a placeholder column

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
            col_metrics, col_equity, col_history, col_trades = st.columns(4) # Added a column for detailed history

            with col_metrics:
                st.download_button(
                    label="Metrics (JSON)",
                    data=metrics_json,
                    # Use values from session state for file naming
                    file_name=f"portfolio_{strategy_name_for_filename}_{symbols_for_filename}_metrics.json",
                    mime="application/json",
                    key='download_portfolio_management_metrics_json', # Unique key
                    on_click=lambda: st.toast("Metrics JSON prepared for download!", icon="💾")
                )

            with col_equity:
                if not equity_curve_df.empty:
                     st.download_button(
                        label="Equity Curve (CSV)",
                        data=equity_curve_df.to_csv(index=False).encode('utf-8'),
                         # Use values from session state for file naming
                        file_name=f"portfolio_{strategy_name_for_filename}_{symbols_for_filename}_equity_curve.csv",
                        mime="text/csv",
                        key='download_portfolio_management_equity_curve_csv', # Unique key
                        on_click=lambda: st.toast("Equity Curve CSV prepared for download!", icon="💾")
                    )
                else:
                     st.info("No equity curve data to download.")

            with col_history:
                 if not detailed_history_df.empty:
                      st.download_button(
                         label="Detailed History (CSV)",
                         data=detailed_history_df.to_csv(index=False).encode('utf-8'),
                         file_name=f"portfolio_{strategy_name_for_filename}_{symbols_for_filename}_detailed_history.csv",
                         mime="text/csv",
                         key='download_portfolio_management_detailed_history_csv', # Unique key
                         on_click=lambda: st.toast("Detailed History CSV prepared for download!", icon="💾")
                      )
                 else:
                      st.info("No detailed history data to download.")


            with col_trades:
                # Check if trades_df was successfully created and does not contain the error column
                if not trades_df.empty and "Error" not in trades_df.columns:
                    st.download_button(
                        label="Trades (CSV)",
                        data=trades_df.to_csv(index=False).encode('utf-8'),
                         # Use values from session state for file naming
                        file_name=f"portfolio_{strategy_name_for_filename}_{symbols_for_filename}_trades.csv",
                        mime="text/csv",
                        key='download_portfolio_management_trades_csv', # Unique key
                        on_click=lambda: st.toast("Trades CSV prepared for download!", icon="💾")
                    )
                elif "Error" in trades_df.columns:
                     st.info(f"Trades data could not be processed for download: {trades_df.loc[0, 'Error']}")
                else:
                     st.info("No trades data to download.")

    else:
         # If button was clicked but no results generated (due to errors or no data)
         st.warning("Backtest did not complete successfully. No results to display.")


else:
    # Clear previous results if backtest button was not clicked in the current run
    # This prevents old results from showing when the page is reloaded or inputs change
    if 'last_portfolio_management_backtest_results' in st.session_state:
        del st.session_state['last_portfolio_management_backtest_results']
    if 'last_portfolio_management_backtest_config' in st.session_state:
         del st.session_state['last_portfolio_management_backtest_config']


# Display warnings below the button if disabled (moved outside the button click block)
# These warnings will persist as long as the conditions are met
# (The warnings inside the button click block are for transient issues during execution)
if not st.session_state.get('portfolio_selected_symbols', []): # Check the session state list
     st.warning("Please select at least one symbol to run the backtest.")
elif strategy_instance is None:
     st.warning("Strategy could not be initialized. Check parameters.")
elif initial_capital <= 0: # Added check for initial capital
     st.warning("Initial capital must be greater than 0.")
elif position_sizing_method != 'Full Equity' and position_sizing_value <= 0:
     st.warning(f"Please enter a valid {position_sizing_method} value.")
