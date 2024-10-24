import streamlit as st
import pandas as pd
from langchain_community.llms import Cohere
import cohere

# Load sentiment data
def load_data(filepath):
    """Load the sentiment analysis data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        st.success("File loaded successfully!")
        return data
    except FileNotFoundError:
        st.error("Error: The file was not found. Please check the file path.")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Initialize Cohere client
def initialize_cohere(api_key):
    """Initialize the Cohere client."""
    return cohere.Client(api_key)

# Recommend products based on user query
def recommend_products(query, data, co):
    """Recommend products based on user input and sentiment data."""
    response = co.generate(prompt=query, model='command-xlarge-nightly')

    # Validate response
    if response.generations:
        search_keywords = response.generations[0].text.strip().split()
        recommendations = data[data['review_description'].str.contains('|'.join(search_keywords), case=False)]
        recommendations = recommendations.sort_values(by='Polarity', ascending=False)
        return recommendations[['product_name', 'Price']].head(1)  # Display top 5 recommendations
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no valid generations

# Set up Streamlit page configuration
st.set_page_config(page_title="Flipkart Product Recommendation", layout="wide")

# Set the background color using HTML
st.markdown(
    """
    <style>
    .main {
        background-color: #E6E6FA;  /* Light purple color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app UI
st.markdown("<h1 style='text-align: center; color: #4B0082;'>📱 Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>Discover the best-rated phones based on user reviews and sentiment analysis!</h3>", unsafe_allow_html=True)

# Create sidebar for additional information
st.sidebar.title("User Guide")
st.sidebar.info("Input your preferences to get personalized phone recommendations. "
                "Examples: best camera phone, lightweight, long battery life, etc.")

st.sidebar.info("⚠️ Note: Please keep your input under 100 words.")

# Create two columns for layout
col1, col2 = st.columns([3, 1])  # 3:1 ratio

with col1:
    # User input section with styled markdown
    st.markdown("<h4 style='color: #800000;'>🎯 Share your preferences:</h4>", unsafe_allow_html=True)

    # Input area for user query
    query = st.text_area("Describe your ideal phone features:", height=100)

    # Load data and initialize Cohere
    data = load_data("sentiment_analysis_flipkart.xls")
    cohere_api_key = 'gFGRs6tOoQpMsKvCP3xKvDdcmecs4PdpFhts136M'
    co = initialize_cohere(cohere_api_key)

    # Button to trigger recommendation
    if st.button("🔍 Recommend"):
        if query:
            with st.spinner("Generating recommendations..."):
                recommended_products = recommend_products(query, data, co)

            if not recommended_products.empty:
                st.markdown("<h3 style='color: #32CD32;'> Recommended Phones:</h3>", unsafe_allow_html=True)
                for idx, row in recommended_products.iterrows():
                    st.markdown(
                        f"""
                        <div style='border: 1px solid #00BFFF; border-radius: 10px; padding: 10px; margin: 10px 0; background-color: #E6E6FA; text-align: center;'>
                            <p style='font-size: 18px; color: #000000;'><b>{row['product_name']}</b> - <span style='color: #FF4500;'>₹{row['Price']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No matching products found. Please try a different query.")
        else:
            st.warning("Please enter a query to get recommendations.")





