import streamlit as st
import os
import sys

# Add parent directory to the path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules (e.g., data sources to check availability)
from data_sources.alpaca_data_source import ALPACA_AVAILABLE
from data_sources.zerodha_data_source import ZERODHA_AVAILABLE


st.title("Settings")

st.header("Data Source Configuration")

st.markdown("""
Configure settings for different data sources and other application parameters.
""")

# --- Data Source Selection (Global Setting) ---
st.subheader("Global Data Source Selection")
st.info("This selection affects which data source is used across all pages (Backtest, Portfolio, Monte Carlo, Paper Trading).")

# Retrieve available data sources from session state (set in app.py)
AVAILABLE_DATA_SOURCES = st.session_state.get('available_data_sources', {})
selected_data_source_name = st.session_state.get('selected_data_source_name', list(AVAILABLE_DATA_SOURCES.keys())[0])

# Use a selectbox to choose the data source
new_selected_data_source_name = st.selectbox(
    "Select Default Data Source",
    list(AVAILABLE_DATA_SOURCES.keys()),
    index=list(AVAILABLE_DATA_SOURCES.keys()).index(selected_data_source_name),
    key='global_data_source_select'
)

# Update session state if the selection changed
if new_selected_data_source_name != selected_data_source_name:
    st.session_state['selected_data_source_name'] = new_selected_data_source_name
    st.toast(f"Default data source set to {new_selected_data_source_name}", icon="✅")
    st.experimental_rerun() # Rerun to apply the change


# --- API Key Configuration (using st.secrets) ---
st.header("API Key Configuration")
st.warning("API keys are sensitive information. Use Streamlit Secrets (`.streamlit/secrets.toml`) for production deployment.")

st.markdown("""
To use data sources that require API keys (like Alpaca or Zerodha), you need to provide your credentials.
For local development, you can create a `.streamlit/secrets.toml` file in your project directory
and add your keys like this:

```toml
ALPACA_API_KEY = "YOUR_ALPACA_API_KEY_ID"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
ZERODHA_API_KEY = "YOUR_ZERODHA_API_KEY"
ZERODHA_ACCESS_TOKEN = "YOUR_ZERODHA_ACCESS_TOKEN" # Note: Access token is temporary and needs to be generated via OAuth flow
```

For deployment on Streamlit Cloud, you can add these secrets directly in the app settings.
""")

# Display current status of API keys from st.secrets
st.subheader("Current API Key Status")

# Alpaca Status
alpaca_key_present = "ALPACA_API_KEY" in st.secrets and bool(st.secrets["ALPACA_API_KEY"])
alpaca_secret_present = "ALPACA_SECRET_KEY" in st.secrets and bool(st.secrets["ALPACA_SECRET_KEY"])
if ALPACA_AVAILABLE:
    if alpaca_key_present and alpaca_secret_present:
        st.success("Alpaca API Keys found in secrets.")
    else:
        st.warning("Alpaca API Keys not found or incomplete in secrets. Alpaca data source may not work.")
else:
     st.info("Alpaca data source library not installed. API keys are not applicable.")


# Zerodha Status
zerodha_key_present = "ZERODHA_API_KEY" in st.secrets and bool(st.secrets["ZERODHA_API_KEY"])
zerodha_access_token_present = "ZERODHA_ACCESS_TOKEN" in st.secrets and bool(st.secrets["ZERODHA_ACCESS_TOKEN"])
if ZERODHA_AVAILABLE:
    if zerodha_key_present and zerodha_access_token_present:
        st.success("Zerodha API Key and Access Token found in secrets.")
        st.warning("Note: Zerodha Access Tokens are temporary. You might need to update `.streamlit/secrets.toml` periodically or implement the full OAuth flow.")
    else:
        st.warning("Zerodha API Key or Access Token not found or incomplete in secrets. Zerodha data source may not work.")
else:
     st.info("Zerodha data source library not installed. API keys are not applicable.")


st.markdown("---")

st.subheader("Other Settings (Placeholder)")
st.info("This section can be used for other application-wide settings in the future.")

