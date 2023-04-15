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
    
    # Load merge.csv
    data2 = pd.read_csv("merge.csv", encoding='utf-8')

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
st.write("## Project Overview")
st.write("This project is to build a recommendation system for Shopee.vn - an e-commerce website. The recommendation system will be based on the content of the product and the user's rating history. The recommendation system will be built using two methods: Content-based Filtering and Collaborative Filtering.")

st.write("## Content-based Filtering")
st.write("Content-based filtering is a recommender system algorithm that recommends items to users based on their similarity to previously consumed items. The algorithm analyzes the attributes of items that a user has interacted with in the past and finds other items with similar attributes. The attributes can include features such as genre, language, author, artist, or any other relevant metadata associated with the items.\nThe content-based filtering algorithm operates on the principle that users who have shown a preference for certain items in the past are likely to be interested in other items with similar attributes. For example, if a user has shown a preference for movies with science fiction genre, the algorithm can recommend other science fiction movies to the user.\nThe content-based filtering algorithm is widely used in e-commerce, media streaming platforms, and online advertising to provide personalized recommendations to users. It is relatively simple to implement and can work well in cases where there is a clear relationship between the attributes of items and user preferences.\nOne of the limitations of content-based filtering is that it tends to recommend items that are similar to what a user has already consumed, which may not always result in serendipitous discovery. Additionally, the algorithm may struggle to recommend items with features that are not explicitly specified in the item metadata.")
st.write("### Cosine Similarity")
st.write("Cosine similarity is a measure of similarity between two vectors that is widely used in content-based filtering algorithms. It measures the cosine of the angle between two vectors in a high-dimensional space.\nIn content-based filtering, cosine similarity is used to measure the similarity between the attributes of two items. The algorithm converts the attributes of each item into a vector, where each dimension of the vector represents a specific attribute. The cosine similarity between two items is then calculated as the dot product of the two vectors divided by the product of their magnitudes.\nCosine similarity ranges from -1 to 1, where a value of 1 indicates that the two items are identical, a value of 0 indicates that the two items are orthogonal, and a value of -1 indicates that the two items are diametrically opposed.\nCosine similarity is widely used in content-based filtering because it is computationally efficient and can work well in cases where the dimensionality of the attribute space is high. Additionally, it can handle cases where the magnitude of the attribute vectors varies widely between items.\nOne limitation of cosine similarity is that it does not take into account the semantic meaning of the attributes. For example, two items may have the same attribute values but be conceptually different, such as a movie and a book with the same title. In such cases, other similarity measures that take into account the semantic meaning of the attributes may be more appropriate.")
st.write("## Collaborative Filtering")
st.write("Collaborative filtering is a recommender system algorithm that recommends items to users based on the preferences and behaviors of other users. The algorithm identifies users who have similar preferences and behaviors and recommends items that these users have consumed or rated highly.\nIn collaborative filtering, the algorithm analyzes the ratings or interactions between users and items and creates a user-item matrix. The matrix contains the ratings or interactions of each user with each item. The algorithm then identifies similar users based on their ratings or interactions and recommends items that similar users have rated highly or consumed.\nCollaborative filtering can be divided into two categories: memory-based and model-based. Memory-based collaborative filtering uses the entire user-item matrix to compute the similarity between users and recommend items. Model-based collaborative filtering uses machine learning algorithms to learn the underlying patterns in the user-item matrix and make recommendations based on these patterns.\nCollaborative filtering can work well in cases where there is a large dataset with many users and items. It can also handle cases where there is no clear relationship between the attributes of items and user preferences.\nOne limitation of collaborative filtering is the cold start problem, where it can be challenging to make recommendations for new items or new users with no or little history of interactions. Additionally, the algorithm may suffer from a popularity bias, where it recommends popular items even if they may not be the best fit for a specific user's preferences.")
# GUI
with st.sidebar:
    choice = option_menu(
        menu_title="Menu",
        options=["Project Objective", "Content-based Filtering", "Collaborative Filtering"],
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
            # Show product rating based on selected user_id
            st.write("Product rating: ", data2.iloc[idx]['rating'])
