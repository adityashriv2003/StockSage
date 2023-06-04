import streamlit as st
from datetime import date
from prophet import Prophet
import yfinance as yf
from prophet.plot import plot_plotly
from plotly import graph_objs as go
# plotly is a free and open-source graphing library for Python used to create 
#interactive graphs


# today's date:
# strftime is used to convert our date into a string format
# The strftime() method returns a string representing date and time using date, time or datetime
today_date = date.today().strftime('%Y-%m-%d')
start_date = '2016-05-08'

# giving our web app a name 
st.title("Stock Prediciton App")
# now we need to provide the user with the options of which stocks to view
stocks = ("AAPL" , "GOOG" , "MSFT" , "GME")
# now we need to provide the user with the option to select which stock to view

selected_stock = st.selectbox("Select company dataset for prediction" , stocks)

# now we need to provide with an interactive slider to the user to select
# the range of the years they want to see the stock prices 
n_years = st.slider("Select year range:" , 1 , 4)
# (1,4) here is the start and end value of the slider from 1 to 4 years is possible 

period = n_years*365

# now we need to load the data from yahoo finance 

@st.cache_data # this is used to save the earlier loaded data of any company , so that we don't have to load it again and again
def load_data(name_stock):
    data = yf.download(name_stock , start_date , today_date)
    # this will give us the data of the given stock comapny from start till today
    # data will now contain a python dataframe 
    # we just need to provide date as the first col
    data.reset_index(inplace = True)
    return data

# creating a streamlit interactive tool to represent the data as text in th start 

data_load_state = st.text("Loading data......")
data = load_data(selected_stock)
data_load_state.text("Loading data process is completed!!!")


st.subheader("Raw Data")
# streamlit can automatically handle the python dataframes , so data.head() will show set of some starting date data
st.write(data.tail())


# fuction to plot the data
def plotRawdata():
    fig = go.Figure()
    # now we create the graph x = date  vs y = prices
    fig.add_trace(go.Scatter(x = data['Date'] , y = data['Open'] , name = "Stock_open_Price"))

    fig.add_trace(go.Scatter(x = data['Date'] , y = data['Close'] , name = "Stock_close_Price"))
    fig.layout.update(title_text = "Time Series Data" , xaxis_rangeslider_visible = True )
    st.plotly_chart(fig)# this is used to plot the graph

plotRawdata()

# forecasting using facebook Prophet
# facebook prophet needs data in a specific format 

df_train = data[['Date' , 'Close']]
# prophet needs the date col as ds and the close price col as y
df_train = df_train.rename(columns = {"Date" : "ds" , "Close" : "y"})

# loading the prophet model 
m = Prophet()
m.fit(df_train)
# now building a dataframe for the future predictions 
future = m.make_future_dataframe(periods = period)

new_text = st.text("Predicting the data.......")
forecast = m.predict(future)
new_text.text("Data prediciton completed!!!")
st.subheader("Forecast Data")

st.write(forecast.tail())


# plotting the forecast
# for plotting the forecast we don't need to define a function we can just use an inbuilt function where we need to pass the model and the forecast data
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# we can also plot the various forecast components by using a built in function plot_components 
# this is a simple graph not a plotly , the inbuilt function is inside the model itself
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)