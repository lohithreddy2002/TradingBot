import streamlit as st
import pandas as pd # Keep necessary imports for global config
import datetime # Keep necessary imports for global config
import sys # Keep necessary imports for global config
import os

# from tradingbot_gemini.data_sources.alphavantage_data_source import AlphaVantageDataSource # Keep necessary imports for global config
# Removed matplotlib.pyplot as it's used within pages now
# Removed streamlit_searchbox import as it's used within pages now
# Removed strategy and data source imports as they are handled by pages or session state

# Add parent directory to the path to allow importing modules
# This might still be useful if pages need to import from directories outside 'pages'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


# --- Configuration ---
st.set_page_config(page_title="Trading Bot",
        page_icon="📈",
        layout="wide")

# --- Available Data Sources and Strategies ---
# Register your data sources and strategies here. Keep this in app.py as it's global.
# Pages will access these from session state.
from data_sources.dummy_data_source import DummyDataSource
from data_sources.yahoo_finance_data_source import YahooFinanceDataSource

from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.rsi_strategy import RsiStrategy
from strategies.macd_strategy import MacdStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.buy_and_hold import BuyAndHoldStrategy
from data_sources.alphavantage_data_source import AlphaVantageDataSource
from data_sources.alpaca_data_source import AlpacaDataSource # Import new data source
from data_sources.zerodha_data_source import ZerodhaDataSource # Import new data source


AVAILABLE_DATA_SOURCES = {
    "Dummy Data Source": DummyDataSource,
    "Yahoo Finance": YahooFinanceDataSource,
    "Alpaca": AlpacaDataSource, # Register Alpaca
    "Zerodha": ZerodhaDataSource, # Register Zerodha
}

AVAILABLE_STRATEGIES = {
    "Moving Average Crossover": MovingAverageCrossover,
    "RSI Strategy": RsiStrategy,
    "MACD Strategy": MacdStrategy,
    "Momentum Strategy": MomentumStrategy,
    "Breakout Strategy": BreakoutStrategy,
    "Buy and Hold": BuyAndHoldStrategy,
}

# Store available data sources and strategies in session state
# This makes them accessible to all pages
st.session_state['available_data_sources'] = AVAILABLE_DATA_SOURCES
st.session_state['available_strategies'] = AVAILABLE_STRATEGIES

# Initialize selected data source name in session state if not already set
if 'selected_data_source_name' not in st.session_state:
    st.session_state['selected_data_source_name'] = list(AVAILABLE_DATA_SOURCES.keys())[0]

home_page = st.Page("pages/home.py", title="Home", icon=":material/add_circle:")
single_backtest_page = st.Page("pages/single_backtest.py", title="Single Backtest", icon=":material/add_circle:")
portfolio_backtest_page = st.Page("pages/portfolio_backtest.py", title="Portfolio Backtest", icon=":material/add_circle:")
compare_strategies_page = st.Page("pages/compare_strategies.py", title="Compare Strategies", icon=":material/add_circle:")
optimize_strategy_page = st.Page("pages/optimize_strategy.py", title="Optimize Strategy", icon=":material/add_circle:")
monte_carlo_page = st.Page("pages/monte_carlo.py", title="Monte Carlo", icon=":material/add_circle:")
settings_page = st.Page("pages/settings.py", title="Settings", icon=":material/add_circle:")
portfolio_management_page = st.Page("pages/portfolio_management.py", title="Portfolio Management", icon=":material/add_circle:")
paper_trading_page = st.Page("pages/paper_trading.py", title="Paper Trading", icon=":material/add_circle:")

pg = st.navigation({
            "Home": [home_page],
            "Backtest": [single_backtest_page, portfolio_backtest_page],
            "Strategy": [compare_strategies_page, optimize_strategy_page,monte_carlo_page],
            "Portfolio Management": [portfolio_management_page],
            "Paper Trading": [paper_trading_page],
            "Tools": [settings_page],
        })
# st.set_page_config(page_title="Trading Strategy Backtester Suite", page_icon=":material/add_circle:")
pg.run()
# --- Main App Content (Optional - appears on all pages above page-specific content) ---
# You can add a header or welcome message here that shows up on every page.
# st.title("Trading Strategy Backtester Suite")
# st.write("Welcome! Select a page from the sidebar.")

# Streamlit will automatically look for files in the 'pages' directory
# and create navigation based on their filenames.
# The code for each page should now reside entirely within its respective file
# (e.g., pages/single_backtest.py, pages/portfolio_backtest.py, etc.).
# The 'show_page()' function is no longer explicitly called from app.py.
# Streamlit executes the code in the selected page file.

