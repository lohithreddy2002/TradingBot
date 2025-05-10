import streamlit as st
import os
import sys

# Add parent directory to the path to allow importing modules
# This ensures that the Home page can also access modules like data_sources and strategies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))



# --- Load Available Data Sources and Strategies (if not already loaded) ---
# This ensures that even if the user lands directly on Home.py,
# the data sources and strategies are available for selection on other pages.
# This logic is also in app.py, but having it here makes the app more robust
# if users share direct links to specific pages.
if 'available_strategies' not in st.session_state:
    st.session_state['available_strategies'] = {}
    try:
        from strategies.moving_average_crossover import MovingAverageCrossover
        from strategies.rsi_strategy import RsiStrategy
        from strategies.macd_strategy import MacdStrategy
        from strategies.breakout_strategy import BreakoutStrategy
        from strategies.momentum_strategy import MomentumStrategy

        st.session_state['available_strategies']['Moving Average Crossover'] = MovingAverageCrossover
        st.session_state['available_strategies']['RSI Strategy'] = RsiStrategy
        st.session_state['available_strategies']['MACD Strategy'] = MacdStrategy
        st.session_state['available_strategies']['Breakout Strategy'] = BreakoutStrategy
        st.session_state['available_strategies']['Momentum Strategy'] = MomentumStrategy
        # Add other strategies here as they are implemented

    except Exception as e:
        st.error(f"Error loading strategies: {e}")
        st.session_state['available_strategies'] = {} # Ensure it's an empty dict on error


# --- Set Default Settings (if not already set) ---
if 'selected_data_source_name' not in st.session_state and st.session_state['available_data_sources']:
     st.session_state['selected_data_source_name'] = list(st.session_state['available_data_sources'].keys())[0]

if 'initial_capital' not in st.session_state:
     st.session_state['initial_capital'] = 100000.0


# --- Landing Page Content ---
st.title("Welcome to the Algorithmic Trading Backtester")

st.markdown("""
This application allows you to backtest various algorithmic trading strategies on historical data.
Explore different strategies, optimize their parameters, and compare their performance to find
the most promising approaches.
""")

st.header("Key Features")

st.markdown("""
- **Single Strategy Backtest:** Test a single strategy on a specific asset over a defined period.
- **Compare Strategies:** Run multiple strategies on the same dataset and compare their performance metrics and equity curves side-by-side.
- **Strategy Analysis:**
    - **Optuna Optimization:** Find the optimal parameters for a strategy based on a chosen performance metric.
    - **Walk-Forward Testing:** Simulate real-world trading by optimizing parameters on historical data and testing them on subsequent out-of-sample data.
    - **Parameter Sensitivity Analysis:** Understand how changing one or two parameters affects a strategy's performance.
- **Portfolio-Level Backtest:** Test a strategy across multiple assets simultaneously to evaluate its performance on a diversified portfolio.
- **Configurable Parameters:** Easily adjust strategy parameters, initial capital, risk management (stop-loss/take-profit), and position sizing.
- **Data Source Integration:** Currently supports Yahoo Finance, Dummy Data, and **Alpha Vantage**. Easily extendable to include other data sources.
- **Interactive Visualizations:** Analyze results with interactive equity curves and other relevant plots.
- **Downloadable Results:** Save backtest metrics, equity curves, and trade logs for further analysis.
""")

st.header("Getting Started")

st.markdown("""
Use the sidebar navigation to explore the different sections of the application:

1.  **Settings:** Configure global settings like the data source and initial capital.
2.  **Single Strategy Backtest:** Start with a basic backtest of one strategy.
3.  **Compare Strategies:** Evaluate multiple strategies against each other.
4.  **Strategy Analysis:** Dive deeper into optimizing and understanding strategy parameters.
5.  **Portfolio-Level Backtest:** Test strategies on a collection of assets.

Choose a data source (now including Alpha Vantage!), select a strategy and asset(s), configure the parameters, and run the backtest or analysis!
""")

st.markdown("---")
st.info("Note: Data fetching relies on the selected data source and may require an internet connection and a valid API key (for sources like Alpha Vantage).")

# Optional: Add a small image or logo if you have one
# st.image("path/to/your/logo.png", width=100)

