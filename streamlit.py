import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Optimize loading data and model by using cache
@st.cache_data
def load_data():
    # 1. Read data
    data = pd.read_csv("Products_ThoiTrangNam_clean.csv", encoding='utf-8')
    data.columns
    # Load merge.csv
    data2 = pd.read_csv("merge.csv", encoding='utf-8')
    data2.columns
    return data, data2

# Optimize running model by using cache
@st.cache_resource
def run_model(data):
    # 2.2. Remove missing values
    tf = TfidfVectorizer(analyzer='word')
    tfidf_matrix = tf.fit_transform(data.name_description_wt)
    model = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return model

#-------------
# Load data
data, data2 = load_data()

# Run model
model = run_model(data)

# GUI
st.title("Data Science Project")
st.write("## Content-based recommendation system for products")

# GUI
with st.sidebar:
    choice = option_menu(
        menu_title="Menu",
        options=["Business Objective", "Content-based Filtering", "Collaborative Filtering"],
        icons=["business", "book", "book"],
        menu_icon="cast",
    )
if choice == 'Business Objective':    
    st.subheader("Business Objective")
elif choice == 'Content-based Filtering':
    st.subheader("Content-based Filtering")
    # Show product name input
    productn = data['product_name']
    def search_products(query):
        results = [product for product in productn if query.lower() in product.lower()]
        return results
    search = st.text_input('Search for a product')
    # if no product is found, show error message
    if not search:
        st.error("Please enter a product in a category")
    else:
        # if no product in suggestions, show error message
        suggestions = search_products(search)
        if not suggestions:
            # Show error message
            st.error("Product name is not in dataset")
        else:
            selected_product = st.selectbox('Select a product', suggestions)
            # Show product name
            st.write("## Product name: ", selected_product)
            # Show recommended products
            st.write("## Recommended products: ")
            # Get index of product
            idx = data[data['product_name'] == selected_product].index[0]
            # Get list of similar products
            similar_products = list(enumerate(model[idx]))
            # Sort list of similar products
            sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)
            # Show random 5 similar products 5
            for i in sorted_similar_products[1:6]:
                # Get index of similar product
                idx = i[0]
                # Get product name   
                product_name = data.iloc[idx]['product_name']
                # Get product price
                product_price = data.iloc[idx]['price']
                # Get product rating
                product_rating = data.iloc[idx]['rating']
                #Align the product name and product price under the product image
                col1, col2 = st.columns(2)
                with col1:
                    # Show product image
                    data[['image']] = data[['image']].astype(str)
                    # If product image is null, show error message
                    if data.iloc[idx]['image'] == 'nan':
                        st.error("Product image is not available")
                    else:
                        st.image(data.iloc[idx]['image'], width=280)
                with col2:
                    # Show product name with bold font and link
                    #st.write("Product name: [{}]({})".format(product_name, data.iloc[idx]['link']))
                    # Show product name with big bold font and link
                    st.write("### [{}]({})".format(product_name, data.iloc[idx]['link']))
                    # Show product price
                    st.write("Product price: {:,} VND".format(product_price))
                    # Show product rating
                    st.write("Product rating: ", product_rating)

elif choice == 'Collaborative Filtering':
    st.subheader("Collaborative Filtering")
    # Select user_id
    user_id = st.selectbox('Select a user_id', data2['user_id'].unique())
    # Show user_id
    st.write("## User_id: ", user_id)
    # Show product_name based on selected user_id
    st.write("## Product_name: ")
    # Get list of product_name based on selected user_id
    product_name = data2[data2['user_id'] == user_id]['product_name'].unique()
    # Align the product name and product price under the product image
    for i in product_name:
        # Get index of product
        idx = data2[data2['product_name'] == i].index[0]
        col1, col2 = st.columns(2)
        with col1:
            # Show product image
            data2[['image']] = data2[['image']].astype(str)
            # If product image is null, show error message
            if data2.iloc[idx]['image'] == 'nan':
                st.error("Product image is not available")
            else:
                st.image(data2.iloc[idx]['image'], width=280)
        with col2:
            # Show product name with bold font and link based on selected user_id
            st.write("Product name: [{}]({})".format(i, data2.iloc[idx]['link']))
