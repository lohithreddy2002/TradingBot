import streamlit as st
import pandas as pd
import datetime
import sys
import os
import time # Import time for simulated live updates
import plotly.graph_objects as go # For plotting equity curve
import plotly.express as px # For plotting allocation pie chart
from streamlit_searchbox import st_searchbox
# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from data_sources.base_data_source import BaseDataSource
# Import specific data sources to check availability and instantiate based on selection
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource
from data_sources.alpaca_data_source import AlpacaDataSource, ALPACA_AVAILABLE # Import Alpaca and availability flag
from data_sources.zerodha_data_source import ZerodhaDataSource, ZERODHA_AVAILABLE # Import Zerodha and availability flag

from strategies.base_strategy import BaseStrategy
# We will simulate the trading logic directly in this page for simplicity initially
# A dedicated PaperTrader class could be developed later for more complex logic.


# --- Helper function for symbol searchbox calling the data source method ---
def search_symbols_paper_trading(search_term: str, **kwargs) -> list[str]:
    """
    Calls the search_symbols method of the currently selected data source for the paper trading page.
    Accepts arbitrary keyword arguments.
    Accesses AVAILABLE_DATA_SOURCES from Streamlit's session state.
    Instantiates the selected data source to call its search method.
    """
    AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
    if not AVAILABLE_DATA_SOURCES:
        print("Error: AVAILABLE_DATA_SOURCES not found in session state.")
        return ["Error: Data sources not loaded."]

    selected_data_source_name = st.session_state.get('selected_data_source_name', list(st.session_state.get('available_data_sources', {}).keys())[0])
    DataSourceClass = AVAILABLE_DATA_SOURCES.get(selected_data_source_name)

    if DataSourceClass:
        try:
            # Instantiate the data source to call search_symbols
            # API keys are handled within the data source's __init__ if needed
            data_source_instance = DataSourceClass()
            if hasattr(data_source_instance, 'search_symbols') and callable(getattr(data_source_instance, 'search_symbols')):
                 return data_source_instance.search_symbols(search_term)
            elif hasattr(data_source_instance, 'get_available_symbols') and callable(getattr(data_source_instance, 'get_available_symbols')):
                  # Fallback if search_symbols is not implemented but get_available_symbols is
                  all_symbols = data_source_instance.get_available_symbols()
                  if "Error fetching" in all_symbols[0] if all_symbols else "": # Check for error message from Zerodha/Alpaca
                       return all_symbols # Return error if fetching failed
                  if not search_term:
                       return all_symbols[:20] # Return first 20 if search term is empty
                  search_term_lower = search_term.lower()
                  return [symbol for symbol in all_symbols if search_term_lower in symbol.lower()][:20]
            else:
                 return [f"Search not available for {selected_data_source_name}"]

        except Exception as e:
            print(f"Error calling search or get_available_symbols for {selected_data_source_name}: {e}")
            return [f"Error searching symbols: {e}"]
    else:
         return [f"Data source '{selected_data_source_name}' not found."]


# --- Session State Initialization for Paper Trading ---
if 'paper_trading_running' not in st.session_state:
    st.session_state['paper_trading_running'] = False
if 'paper_trading_portfolio' not in st.session_state:
    st.session_state['paper_trading_portfolio'] = {
        'cash': 0.0,
        'holdings': {}, # {symbol: quantity}
        'history': pd.DataFrame(columns=['Timestamp', 'Total Value', 'Cash', 'Holdings Value']),
        'trade_log': pd.DataFrame(columns=['Timestamp', 'Symbol', 'Type', 'Quantity', 'Price', 'Total Value', 'Notes']),
        'initial_capital': 0.0
    }
if 'paper_trading_data' not in st.session_state:
    st.session_state['paper_trading_data'] = {} # Store fetched historical data
if 'paper_trading_current_index' not in st.session_state:
    st.session_state['paper_trading_current_index'] = 0 # To simulate iterating through data
if 'paper_trading_config' not in st.session_state:
    st.session_state['paper_trading_config'] = {} # Store configuration used for the current run
if 'paper_trading_selected_symbols' not in st.session_state:
    st.session_state['paper_trading_selected_symbols'] = []
if 'paper_trading_initial_allocation' not in st.session_state:
    st.session_state['paper_trading_initial_allocation'] = {}


# --- Helper function to add a symbol for Paper Trading ---
def add_symbol_to_paper_trading(symbol):
    if symbol and symbol not in st.session_state['paper_trading_selected_symbols']:
        st.session_state['paper_trading_selected_symbols'].append(symbol)
        # Initialize allocation for the new symbol
        st.session_state['paper_trading_initial_allocation'][symbol] = 0.0
        st.toast(f"Added {symbol} to paper trading.", icon="✅")
    elif symbol in st.session_state['paper_trading_selected_symbols']:
        st.warning(f"{symbol} is already selected for paper trading.")
        st.toast(f"{symbol} already added.", icon="⚠️")
    else:
        st.warning("Please select a symbol to add.")
        st.toast("Please select a symbol.", icon="⚠️")


# --- Helper function to remove a symbol from Paper Trading ---
def remove_symbol_from_paper_trading(symbol):
    if symbol in st.session_state['paper_trading_selected_symbols']:
        st.session_state['paper_trading_selected_symbols'].remove(symbol)
        # Remove allocation for the removed symbol
        if symbol in st.session_state['paper_trading_initial_allocation']:
            del st.session_state['paper_trading_initial_allocation'][symbol]
        st.toast(f"Removed {symbol} from paper trading.", icon="🗑️")


# --- Helper function to update portfolio history ---
def update_portfolio_history(portfolio, current_timestamp, current_prices):
    holdings_value = sum(portfolio['holdings'].get(symbol, 0) * current_prices.get(symbol, 0) for symbol in portfolio['holdings'])
    total_value = portfolio['cash'] + holdings_value
    new_row = pd.DataFrame([{
        'Timestamp': current_timestamp,
        'Total Value': total_value,
        'Cash': portfolio['cash'],
        'Holdings Value': holdings_value
    }])
    # Ensure the index is reset before concatenation
    portfolio['history'] = pd.concat([portfolio['history'].reset_index(drop=True), new_row], ignore_index=True)
    return portfolio

# --- Helper function to log a trade ---
def log_trade(portfolio, timestamp, symbol, trade_type, quantity, price, notes=""):
    trade_value = quantity * price
    new_trade = pd.DataFrame([{
        'Timestamp': timestamp,
        'Symbol': symbol,
        'Type': trade_type,
        'Quantity': quantity,
        'Price': price,
        'Total Value': trade_value if trade_type == 'Buy' else -trade_value, # Negative for sell
        'Notes': notes
    }])
    # Ensure the index is reset before concatenation
    portfolio['trade_log'] = pd.concat([portfolio['trade_log'].reset_index(drop=True), new_trade], ignore_index=True)
    return portfolio


# --- Core Paper Trading Simulation Logic ---
def run_paper_trading_simulation(config):
    # Retrieve data and config from session state
    data = st.session_state['paper_trading_data']
    portfolio = st.session_state['paper_trading_portfolio']
    current_index = st.session_state['paper_trading_current_index']

    selected_strategy_name = config['strategy_name']
    strategy_params = config['strategy_parameters']
    initial_capital = config['initial_capital']
    stop_loss_pct = config['stop_loss_pct']
    take_profit_pct = config['take_profit_pct']
    position_sizing_method = config['position_sizing_method']
    position_sizing_value = config['position_sizing_value']
    initial_allocation = config['initial_allocation'] # This will be used for initial setup
    selected_data_source_name = config['data_source_name'] # Get data source name from config

    # Retrieve available strategies from session state
    AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})
    StrategyClass = AVAILABLE_STRATEGIES.get(selected_strategy_name)

    if not StrategyClass:
        st.error(f"Strategy class '{selected_strategy_name}' not found.")
        st.session_state['paper_trading_running'] = False
        return

    try:
        strategy_instance = StrategyClass(**strategy_params)
    except Exception as e:
        st.error(f"Error initializing strategy {selected_strategy_name}: {e}")
        st.session_state['paper_trading_running'] = False
        return

    # --- Simulation Loop ---
    # Iterate through the data index to simulate time passing
    # We assume all symbols have the same index for simplicity in this simulation
    # In a real system, you'd handle differing timestamps.
    # Find the common index across all dataframes
    common_index = None
    for symbol, df in data.items():
         if common_index is None:
              common_index = df.index
         else:
              # Find the intersection of indices
              common_index = common_index.intersection(df.index)

    if common_index is None or common_index.empty:
         st.error("No common dates across selected symbols. Cannot run simulation.")
         st.session_state['paper_trading_running'] = False
         return

    # Sort the common index to ensure chronological order
    common_index = common_index.sort_values()

    # If starting fresh (not resuming), set up initial portfolio
    if current_index == 0:
         portfolio['cash'] = initial_capital
         portfolio['initial_capital'] = initial_capital
         portfolio['holdings'] = {}
         portfolio['history'] = pd.DataFrame(columns=['Timestamp', 'Total Value', 'Cash', 'Holdings Value'])
         portfolio['trade_log'] = pd.DataFrame(columns=['Timestamp', 'Symbol', 'Type', 'Quantity', 'Price', 'Total Value', 'Notes'])

         # Apply initial allocation on the first common date
         first_common_date = common_index[0]
         current_prices_at_start = {symbol: data.get(symbol, pd.DataFrame()).loc[first_common_date].get('Close', None)
                                     for symbol in data if first_common_date in data.get(symbol, pd.DataFrame()).index}

         for symbol, allocation_pct in initial_allocation.items():
              if allocation_pct > 0:
                   first_day_price = current_prices_at_start.get(symbol, None)
                   if first_day_price is not None and first_day_price > 0:
                        allocated_amount = (allocation_pct / 100.0) * initial_capital
                        quantity_to_buy = allocated_amount / first_day_price
                        # Simulate buying
                        portfolio['holdings'][symbol] = portfolio['holdings'].get(symbol, 0) + quantity_to_buy
                        portfolio['cash'] -= allocated_amount # Deduct allocated amount
                        # Log the initial allocation as a 'Buy' trade
                        portfolio = log_trade(portfolio, first_common_date, symbol, 'Buy (Initial)', quantity_to_buy, first_day_price, notes=f"Initial allocation of {allocation_pct}%")
                   else:
                        st.warning(f"Could not apply initial allocation for {symbol}: No data or zero price on start date ({first_common_date.strftime('%Y-%m-%d')}).")

         # Record initial state in history
         portfolio = update_portfolio_history(portfolio, first_common_date, current_prices_at_start)


    # Placeholder for displaying live updates
    live_metrics_placeholder = st.empty()
    equity_chart_placeholder = st.empty()
    allocation_chart_placeholder = st.empty()
    holdings_placeholder = st.empty()
    trade_log_placeholder = st.empty()


    # Simulate processing day by day using the common index
    for i in range(current_index, len(common_index)):
        if not st.session_state['paper_trading_running']:
            st.info("Paper trading simulation stopped.")
            break # Stop if the user clicked the stop button

        current_date = common_index[i]
        # Get data up to the current date for strategy calculation
        data_up_to_today = {symbol: df.loc[:current_date] for symbol, df in data.items() if current_date in df.index or (not df.empty and df.index.max() >= current_date)} # Include data if current_date is in index or latest date is >= current_date

        # Get current day's data (for price and volume) - only for symbols with data on this specific date
        current_day_data = {symbol: df.loc[[current_date]] for symbol, df in data.items() if current_date in df.index}
        current_prices = {symbol: df['Close'].iloc[0] for symbol, df in current_day_data.items()}
        current_volumes = {symbol: df['Volume'].iloc[0] for symbol, df in current_day_data.items()}


        # --- Generate Signals for each asset ---
        signals = {}
        for symbol, df in data_up_to_today.items():
             if not df.empty and current_date in df.index: # Only generate signals if data exists up to today for the symbol
                  try:
                       # Generate signals using data up to the current day
                       signals[symbol] = strategy_instance.generate_signals(df)
                  except Exception as e:
                       st.warning(f"Error generating signals for {symbol} on {current_date}: {e}")
                       signals[symbol] = pd.Series(0, index=df.index) # Default to no signal on error


        # --- Simulate Order Execution ---
        for symbol in st.session_state.get('paper_trading_selected_symbols', []): # Iterate through selected symbols from session state
            if symbol not in current_day_data:
                 # st.warning(f"No data for {symbol} on {current_date}. Skipping signal processing and trading.")
                 continue # Skip if no data for this symbol on this day

            current_price = current_prices.get(symbol)
            if current_price is None or current_price <= 0:
                 # st.warning(f"Invalid price for {symbol} on {current_date}. Skipping signal processing and trading.")
                 continue # Skip if price is invalid

            # Get the latest signal for this symbol (from the signals series up to current_date)
            latest_signal = signals.get(symbol, pd.Series(0)).iloc[-1] if symbol in signals and not signals.get(symbol, pd.Series(0)).empty else 0


            # --- Apply Stop Loss and Take Profit (Simple Global) ---
            # This is a very basic implementation. More sophisticated SL/TP would track per-position.
            if symbol in portfolio['holdings'] and portfolio['holdings'][symbol] > 0:
                 # Find the average entry price for this holding (simplified)
                 # A real system would track entry prices for each trade
                 buy_trades = portfolio['trade_log'][(portfolio['trade_log']['Symbol'] == symbol) & (portfolio['trade_log']['Type'].str.contains('Buy'))]
                 if not buy_trades.empty:
                      # Calculate weighted average entry price
                      average_entry_price = (buy_trades['Quantity'] * buy_trades['Price']).sum() / buy_trades['Quantity'].sum()

                      if stop_loss_pct > 0 and current_price < average_entry_price * (1 - stop_loss_pct):
                           st.info(f"Stop Loss triggered for {symbol} on {current_date.strftime('%Y-%m-%d')}.")
                           # Simulate selling all
                           quantity_to_sell = portfolio['holdings'][symbol]
                           portfolio['cash'] += quantity_to_sell * current_price
                           portfolio['holdings'][symbol] = 0
                           portfolio = log_trade(portfolio, current_date, symbol, 'Sell (SL)', quantity_to_sell, current_price, notes=f"Stop Loss triggered at {stop_loss_pct*100:.2f}%")
                           latest_signal = 0 # Override any other signal to sell

                      elif take_profit_pct > 0 and current_price > average_entry_price * (1 + take_profit_pct):
                           st.info(f"Take Profit triggered for {symbol} on {current_date.strftime('%Y-%m-%d')}.")
                           # Simulate selling all
                           quantity_to_sell = portfolio['holdings'][symbol]
                           portfolio['cash'] += quantity_to_sell * current_price
                           portfolio['holdings'][symbol] = 0
                           portfolio = log_trade(portfolio, current_date, symbol, 'Sell (TP)', quantity_to_sell, current_price, notes=f"Take Profit triggered at {take_profit_pct*100:.2f}%")
                           latest_signal = 0 # Override any other signal to sell


            # --- Execute Trades based on Signals and Position Sizing ---
            # Ensure no consecutive trades of the same type for the same symbol on the same day
            # Check the trade log for the current date and symbol
            trades_today = portfolio['trade_log'][(portfolio['trade_log']['Timestamp'] == current_date) & (portfolio['trade_log']['Symbol'] == symbol)]
            last_trade_type_today = trades_today['Type'].iloc[-1] if not trades_today.empty else None

            # Check for existing position before buying
            has_position = portfolio['holdings'].get(symbol, 0) > 0

            if latest_signal == 1 and portfolio['cash'] > 0 and not has_position and last_trade_type_today != 'Buy': # Buy signal, enough cash, no existing position, and not already bought today
                # Calculate quantity based on position sizing
                quantity_to_buy = 0
                if position_sizing_method == 'Full Equity':
                    quantity_to_buy = portfolio['cash'] / current_price
                elif position_sizing_method == 'Fixed Amount':
                    quantity_to_buy = min(portfolio['cash'], position_sizing_value) / current_price
                elif position_sizing_method == 'Percentage of Equity':
                    total_equity = portfolio['cash'] + sum(portfolio['holdings'].get(s, 0) * current_prices.get(s, 0) for s in portfolio['holdings'])
                    amount_to_allocate = total_equity * position_sizing_value
                    quantity_to_buy = min(portfolio['cash'], amount_to_allocate) / current_price # Ensure we don't spend more cash than available

                if quantity_to_buy * current_price > 0.01: # Avoid tiny trades
                     # Simulate buying
                     cost = quantity_to_buy * current_price
                     portfolio['cash'] -= cost
                     portfolio['holdings'][symbol] = portfolio['holdings'].get(symbol, 0) + quantity_to_buy
                     portfolio = log_trade(portfolio, current_date, symbol, 'Buy', quantity_to_buy, current_price)


            elif latest_signal == -1 and has_position and last_trade_type_today != 'Sell': # Sell signal, holding the asset, and not already sold today
                # Simulate selling all of the holding for this symbol
                quantity_to_sell = portfolio['holdings'][symbol]
                portfolio['cash'] += quantity_to_sell * current_price
                portfolio['holdings'][symbol] = 0 # Sell all
                portfolio = log_trade(portfolio, current_date, symbol, 'Sell', quantity_to_sell, current_price)


        # --- Update Portfolio History ---
        # Need current prices for all symbols with data up to today for accurate history
        current_prices_for_history = {symbol: df['Close'].iloc[-1] for symbol, df in data_up_to_today.items() if not df.empty}
        portfolio = update_portfolio_history(portfolio, current_date, current_prices_for_history)

        # --- Display Live Updates (Optional, can slow down simulation) ---
        # Update placeholders periodically, not on every iteration
        # Calculate the number of days processed so far
        days_processed = i - current_index + 1
        total_days_in_simulation = len(common_index) - current_index

        # Update frequency based on total simulation length to avoid excessive updates
        update_frequency = max(1, total_days_in_simulation // 100) # Update at least every day, or every 1% of simulation

        if days_processed % update_frequency == 0 or i == len(common_index) - 1: # Update periodically or on the last day
             with live_metrics_placeholder.container():
                  st.subheader(f"Live Metrics ({current_date.strftime('%Y-%m-%d')})")
                  current_total_value = portfolio['history']['Total Value'].iloc[-1] if not portfolio['history'].empty else portfolio['initial_capital']
                  current_holdings_value = portfolio['history']['Holdings Value'].iloc[-1] if not portfolio['history'].empty else 0
                  current_cash = portfolio['history']['Cash'].iloc[-1] if not portfolio['history'].empty else portfolio['initial_capital']

                  col_live1, col_live2, col_live3 = st.columns(3)
                  col_live1.metric("Current Total Value", f"${current_total_value:,.2f}")
                  col_live2.metric("Current Cash", f"${current_cash:,.2f}")
                  col_live3.metric("Current Holdings Value", f"${current_holdings_value:,.2f}")

             if not portfolio['history'].empty:
                  with equity_chart_placeholder.container():
                       st.subheader("Simulated Equity Curve")
                       fig = go.Figure()
                       # Ensure history index is datetime for plotting
                       history_for_plot = portfolio['history'].copy()
                       if not isinstance(history_for_plot['Timestamp'].iloc[0], datetime.datetime):
                            history_for_plot['Timestamp'] = pd.to_datetime(history_for_plot['Timestamp'])

                       fig.add_trace(go.Scatter(x=history_for_plot['Timestamp'], y=history_for_plot['Total Value'], mode='lines', name='Portfolio Value'))
                       fig.update_layout(title='Simulated Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value ($)')
                       st.plotly_chart(fig, use_container_width=True)

             # Update allocation chart
             if portfolio['holdings'] or portfolio['cash'] > 0:
                  with allocation_chart_placeholder.container():
                       st.subheader("Current Portfolio Allocation")
                       current_total_value = portfolio['history']['Total Value'].iloc[-1] if not portfolio['history'].empty else portfolio['initial_capital']
                       if current_total_value > 0:
                            # Use current_prices for the latest values
                            current_allocation_data = {symbol: quantity * current_prices.get(symbol, 0) for symbol, quantity in portfolio['holdings'].items() if quantity > 0 and current_prices.get(symbol, 0) is not None}
                            current_allocation_data['Cash'] = portfolio['cash']
                            # Filter out assets/cash with zero value for the chart
                            chart_data = {asset: value for asset, value in current_allocation_data.items() if value > 0}

                            if chart_data:
                                 allocation_df = pd.DataFrame(list(chart_data.items()), columns=['Asset', 'Value'])
                                 # Calculate percentage for display in hover text
                                 allocation_df['Percentage'] = (allocation_df['Value'] / allocation_df['Value'].sum()) * 100
                                 fig_alloc = px.pie(allocation_df, values='Value', names='Asset', title='Current Portfolio Allocation Distribution',
                                                    hover_data=['Percentage']) # Show percentage on hover
                                 st.plotly_chart(fig_alloc, use_container_width=True)
                            else:
                                 st.info("No assets or cash with non-zero value to display in the chart.")
                       else:
                            st.info("Total portfolio value is zero. Cannot display allocation chart.")


             # Update holdings table
             with holdings_placeholder.container():
                  st.subheader("Current Holdings")
                  if portfolio['holdings']:
                       holdings_list = []
                       # Use current_prices for the latest values
                       for symbol, quantity in portfolio['holdings'].items():
                            if quantity > 0:
                                 current_price = current_prices.get(symbol, 0)
                                 holdings_list.append({'Symbol': symbol, 'Quantity': quantity, 'Current Price': current_price, 'Value': quantity * current_price})
                       if holdings_list:
                            holdings_df = pd.DataFrame(holdings_list)
                            st.dataframe(holdings_df, key='paper_trading_holdings_live')
                       else:
                            st.info("No current holdings.")
                  else:
                       st.info("No current holdings.")

             # Update trade log
             with trade_log_placeholder.container():
                  st.subheader("Trade Log")
                  if not portfolio['trade_log'].empty:
                       # Ensure Timestamp is formatted for display
                       trade_log_display = portfolio['trade_log'].copy()
                       if not isinstance(trade_log_display['Timestamp'].iloc[0], datetime.datetime):
                            trade_log_display['Timestamp'] = pd.to_datetime(trade_log_display['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

                       st.dataframe(trade_log_display, key='paper_trading_trade_log_live')
                  else:
                       st.info("No trades executed yet.")

             # Add a small delay to simulate time passing (adjust as needed)
             time.sleep(0.01) # Sleep for 10 milliseconds


        # Store the current index in session state to resume if app reruns
        st.session_state['paper_trading_current_index'] = i + 1

    # --- Simulation Finished ---
    if st.session_state['paper_trading_running']: # Check if simulation finished naturally
         st.session_state['paper_trading_running'] = False
         st.success("Paper trading simulation completed.")
         st.balloons() # Add a little celebration at the end

    # Ensure final state is saved in session state
    st.session_state['paper_trading_portfolio'] = portfolio


# --- UI Elements for the Paper Trading Page ---

st.subheader("Simulation Configuration")

# Use expanders for configuration sections
with st.expander("Data Selection and Capital", expanded=True):
    st.subheader("Data Selection and Capital")
    # Use the global data source selection from settings
    selected_data_source_name = st.session_state.get('selected_data_source_name', list(st.session_state.get('available_data_sources', {}).keys())[0])
    DataSourceClass = st.session_state.get('available_data_sources', {}).get(selected_data_source_name)

    if not DataSourceClass:
         st.error(f"Data source '{selected_data_source_name}' not found.")
         st.stop()

    st.write(f"Using Data Source: **{selected_data_source_name}** (Change in Settings)")

    # Check if the selected data source requires API keys and if they are present
    requires_api_keys = selected_data_source_name in ["Alpaca", "Zerodha"]
    api_keys_present = True
    if selected_data_source_name == "Alpaca":
         if not ALPACA_AVAILABLE:
              st.error("Alpaca data source library not installed. Cannot use Alpaca.")
              api_keys_present = False # Treat as not available if library missing
         elif "ALPACA_API_KEY" not in st.secrets or "ALPACA_SECRET_KEY" not in st.secrets or not st.secrets["ALPACA_API_KEY"] or not st.secrets["ALPACA_SECRET_KEY"]:
              st.warning("Alpaca API Keys not found in Streamlit secrets. Cannot use Alpaca.")
              api_keys_present = False
    elif selected_data_source_name == "Zerodha":
         if not ZERODHA_AVAILABLE:
              st.error("Zerodha data source library not installed. Cannot use Zerodha.")
              api_keys_present = False # Treat as not available if library missing
         elif "ZERODHA_API_KEY" not in st.secrets or "ZERODHA_ACCESS_TOKEN" not in st.secrets or not st.secrets["ZERODHA_API_KEY"] or not st.secrets["ZERODHA_ACCESS_TOKEN"]:
              st.warning("Zerodha API Key or Access Token not found in Streamlit secrets. Cannot use Zerodha.")
              api_keys_present = False
         else:
              st.warning("Remember Zerodha Access Tokens are temporary. Update secrets.toml if needed.")


    if requires_api_keys and not api_keys_present:
         st.error("Please configure your API keys in the Settings page to use this data source.")
         # Disable the start button later based on this flag


    col_date_start, col_date_end = st.columns(2)
    with col_date_start:
        today = datetime.date.today()
        start_date = st.date_input("Simulation Start Date", today - datetime.timedelta(days=365), key='pt_start_date') # Shorter default for paper trading
    with col_date_end:
        end_date = st.date_input("Simulation End Date", today, key='pt_end_date')

    initial_capital = st.number_input("Initial Capital", min_value=1000.0, value=10000.0, step=1000.0, key='pt_initial_capital')


with st.expander("Portfolio Assets and Initial Allocation", expanded=True):
    st.subheader("Portfolio Assets and Initial Allocation")
    st.write("Select symbols for your paper trading portfolio and specify their initial allocation percentage.")
    st.info("The remaining capital will be held as cash.")

    # --- Symbol Selection (Single searchbox + Add Button) ---
    col_search, col_add = st.columns([0.7, 0.3])
    with col_search:
        # Disable search if API keys are required but not present
        is_search_disabled = requires_api_keys and not api_keys_present
        selected_symbol_to_add = st_searchbox(
            search_symbols_paper_trading, # Use the dynamic search function
            label="Search and Select Symbol to Add",
            key="paper_trading_symbol_searchbox_add",
            placeholder="Type to search for symbols...",
            disabled=is_search_disabled # Disable search if API keys are missing
        )
    with col_add:
        # Disable add button if search is disabled or no symbol is selected
        is_add_disabled = is_search_disabled or not selected_symbol_to_add
        st.markdown("<br>", unsafe_allow_html=True) # Add some space to align button
        st.button("Add Symbol", on_click=add_symbol_to_paper_trading, args=(selected_symbol_to_add,), key='pt_add_symbol_button', disabled=is_add_disabled)


    st.markdown("---") # Separator

    st.subheader("Current Portfolio Assets and Allocation (%)")

    # --- Display Selected Symbols and Allocation Inputs ---
    selected_symbols = st.session_state.get('paper_trading_selected_symbols', [])
    initial_allocation = st.session_state.get('paper_trading_initial_allocation', {})
    total_allocated_pct = 0.0

    if selected_symbols:
        col_symbol, col_allocation, col_remove = st.columns([0.5, 0.25, 0.25]) # Adjust widths

        with col_symbol:
             st.markdown("**Symbol**")
        with col_allocation:
             st.markdown("**Allocation (%)**")
        with col_remove:
             st.markdown("**Remove**")


        for i, symbol in enumerate(selected_symbols):
            if symbol not in initial_allocation:
                 initial_allocation[symbol] = 0.0

            with col_allocation:
                current_allocation = initial_allocation.get(symbol, 0.0)
                allocated_pct = st.number_input(
                    " ",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_allocation),
                    step=0.1,
                    format="%.2f",
                    key=f'paper_trading_allocation_{i}_{symbol}' # Unique key
                )
                initial_allocation[symbol] = allocated_pct
                total_allocated_pct += allocated_pct

            with col_remove:
                 st.button("Remove", on_click=remove_symbol_from_paper_trading, args=(symbol,), key=f'pt_remove_symbol_button_{symbol}')

        st.session_state['paper_trading_initial_allocation'] = initial_allocation

        st.markdown("---")

        st.markdown(f"**Total Allocated:** {total_allocated_pct:.2f}%")
        st.markdown(f"**Remaining Cash:** {100.0 - total_allocated_pct:.2f}%")

        if abs(total_allocated_pct - 100.0) > 1e-6:
            st.warning("Total allocation percentage is not 100%. The remaining percentage will be held as cash.")
            st.toast("Total allocation is not 100%.", icon="⚠️")

        if selected_symbols and total_allocated_pct > 0:
             allocation_data = {s: initial_allocation.get(s, 0.0) for s in selected_symbols if initial_allocation.get(s, 0.0) > 0}
             if 100.0 - total_allocated_pct > 1e-6:
                  allocation_data['Cash'] = 100.0 - total_allocated_pct

             if allocation_data:
                  chart_data = {asset: value for asset, value in allocation_data.items() if value > 0}
                  if chart_data:
                       allocation_df = pd.DataFrame(list(chart_data.items()), columns=['Asset', 'Percentage'])
                       fig = px.pie(allocation_df, values='Percentage', names='Asset', title='Initial Portfolio Allocation Distribution',
                                    hover_data=['Percentage'])
                       st.plotly_chart(fig, use_container_width=True)
                  else:
                       st.info("No assets with non-zero allocation to display in the chart.")

    else:
        st.session_state['paper_trading_initial_allocation'] = {}
        st.info("Select symbols above to add them to your portfolio.")


with st.expander("Strategy and Parameters", expanded=True):
    st.subheader("Strategy Configuration")
    # Retrieve available strategies from session state
    AVAILABLE_STRATEGIES = st.session_state.get('available_strategies', {})
    selected_strategy_name = st.selectbox(
        "Select Strategy for Paper Trading",
        list(AVAILABLE_STRATEGIES.keys()),
        key='pt_strategy_select'
    )
    StrategyClass = AVAILABLE_STRATEGIES.get(selected_strategy_name)

    st.subheader("Configure Strategy Parameters")
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
                            current_value = st.session_state.get(f'pt_param_{selected_strategy_name}_{param_name}', default_value)
                            if isinstance(default_value, int):
                                strategy_params[param_name] = st.number_input(param_name, min_value=1, value=int(current_value), step=1, key=f'pt_param_{selected_strategy_name}_{param_name}_int')
                            elif isinstance(default_value, float):
                                strategy_params[param_name] = st.number_input(param_name, value=float(current_value), step=0.1, format="%.2f", key=f'pt_param_{selected_strategy_name}_{param_name}_float')
                            elif isinstance(default_value, bool):
                                param_value = st.checkbox(param_name, value=bool(current_value), key=f'pt_param_{selected_strategy_name}_{param_name}_bool')
                                strategy_params[param_name] = param_value
                            else:
                                strategy_params[param_name] = st.text_input(param_name, str(current_value), key=f'pt_param_{selected_strategy_name}_{param_name}_text')
                            st.session_state[f'pt_param_{selected_strategy_name}_{param_name}'] = strategy_params[param_name]
                else:
                    st.info("This strategy has no configurable parameters.")
            except Exception as e:
                st.warning(f"Could not load parameters for {selected_strategy_name}: {e}")
        else:
             st.info("This strategy has no configurable parameters.")

    # Attempt to instantiate strategy (for validation before running)
    strategy_instance = None
    if StrategyClass:
         try:
              strategy_instance = StrategyClass(**strategy_params)
         except Exception as e:
              st.warning(f"Could not initialize strategy {selected_strategy_name} with provided parameters: {e}")


with st.expander("Risk and Position Management", expanded=True):
    st.subheader("Risk and Position Management")
    col_sl, col_tp = st.columns(2)
    with col_sl:
        pt_stop_loss_pct = st.number_input("Global Stop Loss (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.2f", key='pt_stop_loss_pct') / 100.0
    with col_tp:
        pt_take_profit_pct = st.number_input("Global Take Profit (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key='pt_take_profit_pct') / 100.0

    st.subheader("Position Sizing")
    pt_position_sizing_method = st.selectbox(
        "Method",
        ['Full Equity', 'Fixed Amount', 'Percentage of Equity'],
        key='pt_position_sizing_method'
    )
    pt_position_sizing_value = 0.0
    if pt_position_sizing_method == 'Fixed Amount':
        pt_position_sizing_value = st.number_input("Amount ($)", min_value=1.0, value=100.0, step=10.0, key='pt_position_sizing_amount') # Smaller default for paper trading
    elif pt_position_sizing_method == 'Percentage of Equity':
        pt_position_sizing_value = st.number_input("Percentage (%)", min_value=0.1, max_value=100.0, value=5.0, step=0.1, format="%.2f", key='pt_position_sizing_percentage') / 100.0 # Smaller default for paper trading


# --- Start/Stop Paper Trading Buttons ---
st.markdown("---")

col_start_stop, col_reset = st.columns(2)

with col_start_stop:
    # Disable start button if already running, or if config is invalid, or if API keys are missing
    is_start_disabled = st.session_state['paper_trading_running'] or not selected_symbols or strategy_instance is None or initial_capital <= 0 or (pt_position_sizing_method != 'Full Equity' and pt_position_sizing_value <= 0) or (requires_api_keys and not api_keys_present)

    if st.button("Start Paper Trading", key='start_paper_trading_button', disabled=is_start_disabled):
        if start_date >= end_date:
            st.error("Error: Start date must be before end date.")
            st.toast("Error: Start date must be before end date.", icon="❌")
        elif not selected_symbols:
             st.warning("Please select at least one symbol to start paper trading.")
             st.toast("Please select at least one symbol.", icon="⚠️")
        elif strategy_instance is None:
             st.warning("Strategy could not be initialized. Check parameters.")
             st.toast("Strategy initialization failed.", icon="❌")
        elif initial_capital <= 0:
             st.warning("Initial capital must be greater than 0.")
             st.toast("Invalid initial capital.", icon="⚠️")
        elif pt_position_sizing_method != 'Full Equity' and pt_position_sizing_value <= 0:
             st.warning(f"Please enter a valid {pt_position_sizing_method} value.")
             st.toast(f"Invalid {pt_position_sizing_method} value.", icon="⚠️")
        elif requires_api_keys and not api_keys_present:
             st.warning("API keys are missing for the selected data source. Please configure them in Settings.")
             st.toast("API keys missing.", icon="❌")
        else:
            # --- Initialize Paper Trading State ---
            st.session_state['paper_trading_running'] = True
            st.session_state['paper_trading_current_index'] = 0 # Start from the beginning of data
            st.session_state['paper_trading_portfolio'] = { # Reset portfolio state
                'cash': 0.0, # Will be set to initial_capital in simulation
                'holdings': {},
                'history': pd.DataFrame(columns=['Timestamp', 'Total Value', 'Cash', 'Holdings Value']),
                'trade_log': pd.DataFrame(columns=['Timestamp', 'Symbol', 'Type', 'Quantity', 'Price', 'Total Value', 'Notes']),
                'initial_capital': initial_capital # Store initial capital
            }
            st.session_state['paper_trading_config'] = { # Store current configuration
                'strategy_name': selected_strategy_name,
                'strategy_parameters': strategy_params,
                'initial_capital': initial_capital,
                'stop_loss_pct': pt_stop_loss_pct,
                'take_profit_pct': pt_take_profit_pct,
                'position_sizing_method': pt_position_sizing_method,
                'position_sizing_value': pt_position_sizing_value,
                'initial_allocation': st.session_state.get('paper_trading_initial_allocation', {}), # Use the allocation from session state
                'data_source_name': selected_data_source_name # Store data source name in config
            }

            # --- Fetch Data Before Starting Simulation ---
            st.info(f"Fetching data for {', '.join(selected_symbols)} from {start_date} to {end_date} using {selected_data_source_name}...")
            st.toast(f"Fetching data for {', '.join(selected_symbols)}...", icon="⏳")
            data = {}
            fetch_errors = []
            DataSourceClass = st.session_state.get('available_data_sources', {}).get(selected_data_source_name)

            if DataSourceClass:
                 try:
                     # Instantiate the selected data source
                     data_source_instance = DataSourceClass()

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

                 except Exception as e:
                      st.error(f"Error initializing data source: {e}")
                      st.toast("Data source error.", icon="❌")
                      st.session_state['paper_trading_running'] = False
                      st.rerun() # Use st.rerun()
                      #  return # Exit the function after triggering rerun


            if fetch_errors:
                for error in fetch_errors:
                    st.error(error)
            if not data:
                 st.error("No data available for paper trading after fetching.")
                 st.toast("No data available.", icon="❌")
                 st.session_state['paper_trading_running'] = False
                 st.rerun() # Use st.rerun()
                #  return # Exit the function after triggering rerun


            if data:
                 st.session_state['paper_trading_data'] = data # Store fetched data in session state
                 st.success("Data fetched successfully. Starting simulation...")
                 st.toast("Data fetched! Starting simulation...", icon="✅")
                 # Trigger rerun to start the simulation loop
                 st.rerun() # Use st.rerun()
                #  return # Exit the function after triggering rerun
            else:
                 st.error("Failed to fetch data. Cannot start simulation.")
                 st.toast("Failed to fetch data.", icon="❌")
                 st.session_state['paper_trading_running'] = False
                 st.rerun() # Use st.rerun()
                 #  return # Exit the function after triggering rerun


with col_start_stop:
    # Disable stop button if not running
    is_stop_disabled = not st.session_state['paper_trading_running']

    if st.button("Stop Paper Trading", key='stop_paper_trading_button', disabled=is_stop_disabled):
        st.session_state['paper_trading_running'] = False
        st.info("Stopping simulation...")
        st.toast("Stopping simulation...", icon="✋")
        # No rerun needed here, the loop check will handle stopping


with col_reset:
     # Reset button to clear state
     if st.button("Reset Paper Trading", key='reset_paper_trading_button'):
          st.session_state['paper_trading_running'] = False
          st.session_state['paper_trading_portfolio'] = {
               'cash': 0.0,
               'holdings': {},
               'history': pd.DataFrame(columns=['Timestamp', 'Total Value', 'Cash', 'Holdings Value']),
               'trade_log': pd.DataFrame(columns=['Timestamp', 'Symbol', 'Type', 'Quantity', 'Price', 'Total Value', 'Notes']),
               'initial_capital': 0.0
          }
          st.session_state['paper_trading_data'] = {}
          st.session_state['paper_trading_current_index'] = 0
          st.session_state['paper_trading_config'] = {}
          st.session_state['paper_trading_selected_symbols'] = [] # Also clear selected symbols
          st.session_state['paper_trading_initial_allocation'] = {} # Also clear allocation
          st.toast("Paper trading state reset.", icon="🔄")
          st.rerun() # Use st.rerun()


# --- Display Live Results if Simulation is Running or has Completed ---
if st.session_state['paper_trading_running'] or (not st.session_state['paper_trading_running'] and st.session_state['paper_trading_current_index'] > 0):
    st.markdown("---")
    st.header("Paper Trading Simulation Live Results")

    # Retrieve current state from session state
    current_portfolio = st.session_state['paper_trading_portfolio']
    current_config = st.session_state['paper_trading_config']
    current_data = st.session_state['paper_trading_data']

    # Re-run the simulation loop if it's supposed to be running
    if st.session_state['paper_trading_running'] and current_data:
         # Pass the stored config to the simulation function
         run_paper_trading_simulation(current_config)
    elif not st.session_state['paper_trading_running'] and st.session_state['paper_trading_current_index'] > 0:
         # Display final results if simulation finished
         st.info("Simulation finished. Displaying final results.")

         # Display final metrics
         st.subheader("Final Performance Metrics")
         final_portfolio_value = current_portfolio['history']['Total Value'].iloc[-1] if not current_portfolio['history'].empty else current_portfolio['initial_capital']
         total_return = (final_portfolio_value - current_portfolio['initial_capital']) / current_portfolio['initial_capital'] if current_portfolio['initial_capital'] > 0 else 0

         # Simple annualized return calculation (assumes daily data and full simulation period)
         annualized_return = 0
         if not current_portfolio['history'].empty and current_portfolio['initial_capital'] > 0:
              start_date_hist = pd.to_datetime(current_portfolio['history']['Timestamp'].iloc[0])
              end_date_hist = pd.to_datetime(current_portfolio['history']['Timestamp'].iloc[-1])
              time_delta_years = (end_date_hist - start_date_hist).days / 365.25
              if time_delta_years > 0:
                   annualized_return = (1 + total_return)**(1 / time_delta_years) - 1


         col_final1, col_final2, col_final3 = st.columns(3)
         col_final1.metric("Initial Capital", f"${current_portfolio['initial_capital']:,.2f}")
         col_final2.metric("Final Portfolio Value", f"${final_portfolio_value:,.2f}")
         col_final3.metric("Total Return", f"{total_ret:.2%}") # Use total_ret variable name from backtest results
         if annualized_return != 0: # Only show annualized if calculated
              st.metric("Annualized Return", f"{annualized_return:.2%}")


         # Display final equity curve
         st.subheader("Final Simulated Equity Curve")
         if not current_portfolio['history'].empty:
              fig = go.Figure()
              history_for_plot = current_portfolio['history'].copy()
              if not isinstance(history_for_plot['Timestamp'].iloc[0], datetime.datetime):
                   history_for_plot['Timestamp'] = pd.to_datetime(history_for_plot['Timestamp'])
              fig.add_trace(go.Scatter(x=history_for_plot['Timestamp'], y=history_for_plot['Total Value'], mode='lines', name='Portfolio Value'))
              fig.update_layout(title='Simulated Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value ($)')
              st.plotly_chart(fig, use_container_width=True)
         else:
              st.info("No equity history to display.")


         # Display final allocation chart
         st.subheader("Final Portfolio Allocation")
         if current_portfolio['holdings'] or current_portfolio['cash'] > 0:
              # Use the last available prices from the fetched data for final allocation calculation
              latest_prices = {symbol: data.get(symbol, pd.DataFrame()).iloc[-1].get('Close', 0) for symbol in current_portfolio['holdings'] if not data.get(symbol, pd.DataFrame()).empty}
              final_allocation_data = {symbol: quantity * latest_prices.get(symbol, 0) for symbol, quantity in current_portfolio['holdings'].items() if quantity > 0 and latest_prices.get(symbol, 0) is not None}
              final_allocation_data['Cash'] = current_portfolio['cash']
              chart_data = {asset: value for asset, value in final_allocation_data.items() if value > 0}

              if chart_data:
                   allocation_df = pd.DataFrame(list(chart_data.items()), columns=['Asset', 'Value'])
                   allocation_df['Percentage'] = (allocation_df['Value'] / allocation_df['Value'].sum()) * 100
                   fig_alloc = px.pie(allocation_df, values='Value', names='Asset', title='Final Portfolio Allocation Distribution',
                                      hover_data=['Percentage'])
                   st.plotly_chart(fig_alloc, use_container_width=True)
              else:
                   st.info("No assets or cash with non-zero value to display in the final allocation chart.")
         else:
              st.info("Final total portfolio value is zero. Cannot display allocation chart.")


         # Display final holdings
         st.subheader("Final Holdings")
         if current_portfolio['holdings']:
              final_holdings_list = []
              # Need latest prices to calculate final value
              latest_prices = {symbol: data.get(symbol, pd.DataFrame()).iloc[-1].get('Close', 0) for symbol in current_portfolio['holdings'] if not data.get(symbol, pd.DataFrame()).empty}
              for symbol, quantity in current_portfolio['holdings'].items():
                   if quantity > 0:
                        latest_price = latest_prices.get(symbol, 0)
                        final_holdings_list.append({'Symbol': symbol, 'Quantity': quantity, 'Final Price': latest_price, 'Value': quantity * latest_price})
              if final_holdings_list:
                   final_holdings_df = pd.DataFrame(final_holdings_list)
                   st.dataframe(final_holdings_df, key='paper_trading_holdings_final')
              else:
                   st.info("No final holdings.")
         else:
              st.info("No final holdings.")


         # Display final trade log
         st.subheader("Final Trade Log")
         if not current_portfolio['trade_log'].empty:
              trade_log_display = current_portfolio['trade_log'].copy()
              if not isinstance(trade_log_display['Timestamp'].iloc[0], datetime.datetime):
                   trade_log_display['Timestamp'] = pd.to_datetime(trade_log_display['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
              st.dataframe(trade_log_display, key='paper_trading_trade_log_final')
         else:
              st.info("No trades executed during the simulation.")


    # If simulation is running, display live placeholders (these are updated within the simulation loop)
    # The placeholders are defined within the run_paper_trading_simulation function
    # and will be populated during the simulation.
    # This section just ensures they exist in the UI layout when the simulation is active.
    elif st.session_state['paper_trading_running']:
         st.subheader("Live Metrics")
         st.empty() # Placeholder for live metrics
         st.subheader("Simulated Equity Curve")
         st.empty() # Placeholder for equity curve chart
         st.subheader("Current Portfolio Allocation")
         st.empty() # Placeholder for allocation chart
         st.subheader("Current Holdings")
         st.empty() # Placeholder for holdings table
         st.subheader("Trade Log")
         st.empty() # Placeholder for trade log table

