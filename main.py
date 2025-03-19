import streamlit as st
from datetime import date

import yfinance as yf

import prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG","MSFT", "^FTSE", "TSLA", "BTC-USD", "NVDA")

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of Prediction:", 1 , 4)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True) # place data in very first column
    return data

data_load_state = st.text("Load data....")
data = load_data(selected_stock)
data_load_state.text("Loading data.. Done")