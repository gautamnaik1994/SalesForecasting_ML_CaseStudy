import streamlit as st
from services.individual_store import forecast
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")
st.title("Individual Store Forecasting")
st.image("https://source.unsplash.com/random/800x400")
# train_data = pd.read_parquet("deploy/data/train_data.parquet")

with st.form(key='forecast_form', border=False):
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")

    with col1:
        store_id = st.number_input(
            "Select a Store ID between 1 to 365", min_value=1, max_value=365, value=3)

    with col2:
        forecast_btn = st.form_submit_button(
            "Forecast", use_container_width=True, type="primary")

if forecast_btn:
    with st.spinner("Forecasting Sales..."):
        forecast_data = forecast(
            Store_id=store_id, days=60)
        forecast_data.index = forecast_data["Date"]
        fig = px.line(forecast_data, x=forecast_data.index, y="Sales",

                      color="Type",
                      color_discrete_map={
                          "No": "#03a9f4",
                          "Yes": "#4caf50"
                      })
        fig.update_xaxes(title_text="Date")
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            height=600,
            title={
                'text': f"Sales Forecast for Store {store_id}",
                'font': {
                    'size': 24
                }
            }
        )
        st.plotly_chart(fig)

st.subheader("Technical Details")

details = """
- **Model Used**: LightGBM
- **Forecasting Method**: Recursive Forecasting
  This method uses the last 30 days of sales data to forecast the next day's
  sales. The forecasted sales are then used to forecast the next day's sales.
- **Feature Engineering**
  - Target Encoding: Store_id
  - Lag Features: Sales
  - Rolling Window Features: Sales

"""

st.markdown(details)
