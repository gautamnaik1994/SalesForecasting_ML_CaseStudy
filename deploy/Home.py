import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(layout="wide")

# Set up the page header title
st.title("My Streamlit App")

# Display an image
# Update with your image path
# image_path = os.path.join("path_to_your_image", "image.png")
st.image("https://www.streamlit.io/images/brand/streamlit-mark-color.png", width=200)
