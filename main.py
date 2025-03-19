import streamlit as st
from datetime import date

import yfinance as yf

import prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App (Test version just for ethan)")

stocks = ("AAPL", "GOOG","MSFT", "^FTSE", "TSLA", "BTC-USD", "NVDA")

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of Prediction:", 1 , 4)
period = n_years * 365

@st.cache  # cache data of stock
def load_data(ticker):
    data = yf.download(ticker,START,TODAY) # already in panda
    data.reset_index(inplace=True) # place data in very first column
    return data

data_load_state = st.text("Load data....")
data = load_data(selected_stock)
data_load_state.text("Loading data.. Done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting of the data
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = prophet()
m.fit(df_train)
future = m.make_future_data_frame(periods = period)
forecast = m.predict(future)

st.subheader('Predicted Data')
st.write(forecast.tail())

st.write('Predicted Data Chart')
figpred = plot_plotly(m, forecast)
st.plotly_chart(figpred)

st.write('forecast components')
figpred1 = m.plot_components(forecast)
st.write(figpred1)
