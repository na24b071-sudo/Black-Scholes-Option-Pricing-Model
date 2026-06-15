import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from Important import black_scholes
import yfinance as yf
from datetime import datetime, timedelta

st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

def heatmap(k,t,r,spot_range, vol_range):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i,j] = black_scholes(spot,k,t,r,vol,'CALL')
            put_prices[i,j] = black_scholes(spot,k,t,r,vol,'PUT')

    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

@st.cache_data
def get_option_timeseries(ticker, strike_price, time_to_expiry, r, vol, option_type='CALL'):
    stock = yf.Ticker(ticker)
    end = datetime.today()
    start = end - timedelta(days=int(time_to_expiry * 365 * 1.5))  # fetch more days for better visuals

    hist = stock.history(start=start, end=end)
    hist = hist[hist['Close'].notnull()]

    end = end.replace(tzinfo=None)
    hist.index = hist.index.tz_localize(None)

    hist['T'] = (time_to_expiry * 365 - (end - hist.index).days) / 365  # decreasing T
    hist['Option_Price'] = hist.apply(
        lambda row: black_scholes(row['Close'], strike_price, max(row['T'], 1e-5), r, vol, option_type),
        axis=1
    )
    return hist[['Close', 'Option_Price']]

st.title("Black-Scholes Model")

st.sidebar.title("Adjust your parameters:")

s = st.sidebar.number_input('Underlying Price: ', value=100)
k = st.sidebar.number_input('Strike Price: ', value=100)
t = st.sidebar.number_input('Time till maturity (years): ', value=1)
r = st.sidebar.number_input('Risk Free Interest Rate: ', value=0.05)
v = st.sidebar.number_input('Volatility: ', value=0.3)

st.markdown('---')

st.sidebar.title('Adjust your Heatmap Parameters:')

spot_min, spot_max = st.sidebar.slider('Spot Price Range:', 0.01, 200.0, (50.0,150.0), 0.01)
vol_min, vol_max = st.sidebar.slider('Volatility Price Range:', 0.01, 1.0, (0.0,1.0), 0.01)

spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)

df = pd.DataFrame({
    "Underlying Price" : [s],
    "Strike Price" : [k],
    "Time till Maturity (years)" : [t],
    "Risk-Free Interest Rate" : [r],
    "Volatility" : [v]
})

st.table(df)

col1, col2 = st.columns(2)
call = black_scholes(s, k, t, r, v, 'CALL')
put = black_scholes(s, k, t, r, v, 'PUT')

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('')

st.title('Options Price - Interactive Heatmap')

call_heatmap, put_heatmap = st.columns(2)

with call_heatmap:
    st.header('Call Heatmap')
    heatmap_fig_call, _ = heatmap(k,t,r, spot_range, vol_range)
    st.pyplot(heatmap_fig_call)

with put_heatmap:
    st.header('Put Heatmap')
    _, heatmap_fig_put = heatmap(k,t,r, spot_range, vol_range)
    st.pyplot(heatmap_fig_put)

st.markdown('')

st.title("Option Price Time Series Visualization")

ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")

if st.button("Show Time Series"):
    with st.spinner("Fetching data..."):
        data = get_option_timeseries(ticker, k, t, r, v, 'CALL')
    
    st.line_chart(data.rename(columns={"Close": "Stock Price", "Option_Price": "Call Option Price"}))
