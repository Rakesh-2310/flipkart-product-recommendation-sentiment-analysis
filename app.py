import streamlit as st
import pandas as pd
import pickle
import requests
from PIL import Image
from io import BytesIO

# Load the pickle file containing LangChain results
try:
    with open("recommended_phones.pkl", 'rb') as f:
        df = pickle.load(f)
except FileNotFoundError:
    st.error("Pickle file not found. Please check the file path.")
    st.stop()

# Add a column with mobile prices and brands
price_data = {
    "product_id": ['iPhone 12', 'vivo T3 Ultra', 'Google Pixel 8', 'IQOO Neo9 Pro', 'SAMSUNG Galaxy S23 5G'],
    "Price": [30999, 35999, 38999, 38999, 39999],
    "brand": ['iPhone', 'vivo', 'Google', 'IQOO', "SAMSUNG Galaxy"]
}

price_df = pd.DataFrame(price_data)
ds = df.merge(price_df, on='product_id', how='left')

st.title("Mobile Phone Recommendations")
st.write("### Based on sentiment analysis from reviews:")

# Sidebar for brand selection
brand_options = ["All"] + list(ds['brand'].unique())  # Add "All" option
selected_brand = st.sidebar.selectbox("Select Brand:", brand_options)

# Sidebar for price range selection
price_filter = st.sidebar.slider("Select Price Range (in INR):", min_value=min(price_data["Price"]), max_value=max(price_data["Price"]), value=(min(price_data["Price"]), max(price_data["Price"])))

# Filter based on selected brand and price
if selected_brand != "All":
    filtered_phones = ds[
        (ds['brand'] == selected_brand) & 
        (ds['Price'] >= price_filter[0]) & 
        (ds['Price'] <= price_filter[1])
    ]
else:
    filtered_phones = ds[
        (ds['Price'] >= price_filter[0]) & 
        (ds['Price'] <= price_filter[1])
    ]

# Show filtered results
if not filtered_phones.empty:
    for _, row in filtered_phones.iterrows():
        prompt = f"""
        Based on sentiment analysis, we recommend the following mobile phone:

        **Model**: {row['product_id']}
        - Positive review Count: {row['average_compound_sentiment']}
        - Price: ₹{row['Price']}
        """

        st.write(prompt)
else:
    st.write("No mobile phones available in this brand and price range.")
