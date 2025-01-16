import streamlit as st
from services.individual_region import forecast
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")
st.title("Individual Regions Forecasting")
st.image("https://source.unsplash.com/random/800x400")

with st.form(key='forecast_form', border=False):
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")

    with col1:
        region_code = st.selectbox(
            "Select a Region",
            ("R1", "R2", "R3", "R4"),
        )

    with col2:
        forecast_btn = st.form_submit_button(
            "Forecast", use_container_width=True, type="primary")

if forecast_btn:
    with st.spinner("Forecasting Sales..."):
        forecast_data = forecast(
            Region_Code=region_code, days=60)
        forecast_data.index = forecast_data["Date"]
        fig = px.line(forecast_data, x=forecast_data.index, y="Total_Sales",

                      color="Type",
                      color_discrete_map={
                          "Current": "#03a9f4",
                          "Forecasted": "#4caf50"
                      })
        fig.update_xaxes(title_text="Date")
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            height=600,
            title={
                'text': f"Sales Forecast for Region {region_code}",
                'font': {
                    'size': 24
                }
            }
        )
        st.plotly_chart(fig)

st.subheader("Technical Details")

details = """
- **Model Used**: Prophet
- **Forecasting Method**: Batch Forecasting
"""

st.markdown(details)
