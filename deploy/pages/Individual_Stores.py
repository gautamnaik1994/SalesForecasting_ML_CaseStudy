import streamlit as st
from services.individual_store import forecast
import plotly.express as px


# Page title
st.title("Individual Stores")
st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", width=200)

# Store selection
store_id = st.number_input("Store ID", min_value=1, max_value=365, value=3)

if st.button("Forecast"):

    forecast_data = forecast(Store_id=store_id, days=31)
    print(forecast_data)
    st.dataframe(forecast_data)
    forecast_data.index = forecast_data["Date"]
    fig = px.line(forecast_data, x=forecast_data.index, y="Sales",
                  color="predicted", title=f"Sales Forecast for Store {store_id}")
    st.plotly_chart(fig)
