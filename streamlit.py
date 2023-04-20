import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from streamlit_image_select import image_select
import warnings
import re

warnings.filterwarnings("ignore")

# Optimize loading data and model by using cache
@st.cache_data
def load_data():
    # 1. Read data
    data = pd.read_csv("Products_ThoiTrangNam_clean.csv", encoding='utf-8')
    
    # Load merge.csv
    data2 = pd.read_csv("merge.csv", encoding='utf-8')

    # Load shopee_1.jpg
    #image1 = Image.open("shopee_1.jpg")

    return data, data2

# Optimize running model by using cache
@st.cache_resource
def clean_text(text):
    text_clean = str(text).lower()
    # Loai bo thong tin lien quan den chi tiet nhu xuat xu, danh muc, kho hang
    if "THÔNG TIN SẢN PHẨM\n" in text_clean:
        text_clean = text_clean[text_clean.index("THÔNG TIN SẢN PHẨM \n"):]
    elif "MÔ TẢ SẢN PHẨM\n" in text_clean:
        text_clean = text_clean[text_clean.index("MÔ TẢ SẢN PHẨM\n"):]
    elif "\n\n" in text_clean:
        text_clean = text_clean[text_clean.index("\n\n"):]
    elif "\nGửi từ\n" in text_clean:
        text_clean = text_clean[text_clean.index("\nGửi từ\n"):]

    # loai bo phan size
    text_clean = re.sub(r"\nsize[^\n]*", "", text_clean)
    # loai bo cac hashtag
    text_clean = re.sub(r"#[^#]*", "", text_clean)
    # loai bo cac ky tu khong hop le
    text_clean = re.sub(r"\n", " ", text_clean)
    # loai bo cac ky tu khong phai la chu hoac so
    text_clean = re.sub(r"[^\w\s]+", " ", text_clean)
    # loai bo cac tu khong can thiet
    text_clean = re.sub(r'\b[smlx]{1,4}\b', '', text_clean)
    text_clean = re.sub(r'\b\d+[kgcm]*\b', '', text_clean)
    # loai bo khang trang thua
    text_clean = re.sub('\s+', ' ', text_clean)
    # loai bo cac ky tu don
    text_clean = re.sub(r"\b[a-zA-Z]\b", "", text_clean)
    # loai bo cac chu so
    text_clean = re.sub(r"\d+", "", text_clean)
    # xu ly khoang trang thua
    text_clean = re.sub(r"\s+", " ", text_clean)
    return text_clean

def find_similar_products(keyword, product_descriptions):
    # Create a TfidfVectorizer object to convert product descriptions into a numerical representation
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the product descriptions into a tf-idf matrix
    tfidf_matrix = vectorizer.fit_transform(product_descriptions)
    
    # Convert the keyword into a tf-idf vector
    keyword_tfidf = vectorizer.transform([keyword])
    
    # Calculate cosine similarity between the keyword and all product descriptions
    cosine_similarities = cosine_similarity(keyword_tfidf, tfidf_matrix).flatten()
    
    # Sort the indices of the products based on their cosine similarity with the keyword
    similar_product_indices = np.argsort(cosine_similarities)[::-1][:10]
    
    # Return the top 10 most similar products
    return [product_descriptions[i] for i in similar_product_indices]

# Get the product names corresponding to the product descriptions
def get_similar_product_names(similar_products):
    similar_product_names = []
    for product in similar_products:
        similar_product_names.append(data.loc[data['name_description_wt'] == product]['product_name'].tolist()[0])
    return similar_product_names

def run_model(data):
    tf = TfidfVectorizer(analyzer='word')
    tfidf_matrix = tf.fit_transform(data.name_description_wt)
    model = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return model

#-------------
# Load data
data, data2 = load_data()
data[['image']] = data[['image']].astype(str)

# Def star rating
def star_rating(rating):
    if rating == 5:
        return "★★★★★"
    elif rating == 4:
        return "★★★★"
    elif rating == 3:
        return "★★★"
    elif rating == 2:
        return "★★"
    elif rating == 1:
        return "★"
    else:
        return "☆"

# Run model
model = run_model(data)

# GUI
st.title("Data Science Project")

with st.sidebar:
    choice = option_menu(
        menu_title="Menu",
        options=["Project Overview", "Content-based Filtering_1", "Content-based Filtering_2", "Collaborative Filtering"],
        icons=["book", "book", "book", "book"],
        menu_icon="cast",
    )
if choice == 'Project Overview':    
    st.write("### Project Objective")
    st.image("https://golocad.com/wp-content/uploads/2022/11/shopee-logistics-engine.webp", width=700)
    st.write("This project is to build a recommendation system for Shopee.vn, an e-commerce website. The recommendation system will be based on the content of the product and the user's rating history.")
    st.write("The recommendation system will be built using two methods: Content-based Filtering and Collaborative Filtering.")
    st.write("1. Content-based Filtering: recommends items to users based on their similarity to the previously searched items.")
    st.write("2. Collaborative Filtering: recommends items to users based on the preferences and behaviors of other users. The algorithm identifies users who have similar preferences and behaviors and recommends items that these users have consumed or rated highly.")
    st.image("https://co-libry.com/wp-content/uploads/2020/05/Recommendation-engines-Co-libry-E-commerce-1.png", width=700)
    
    st.write("### Case 01: Content-based Filtering")
    st.write("#### Understand the Dataset")
    st.write("The data used in this case is the product data of Shopee.vn. The data contains 49,653 products with 9 attributes: product_id, product_name, price, rating, link, category, sub-category, image, description.")
    st.write("The data is collected from the Shopee.vn.")
    
    st.write("#### Preprocessing data")
    st.write("The data is then cleaned and preprocessed using the Python library Pandas, underthesea")
    st.write("1. Process raw data: Remove unnecessary phrases, Lowercase all words, Remove special characters, Replace emoji/teencode with text, punctuations with space, Replace multiple spaces with single space,.etc")
    st.write("2. Standardize Vietnamese unicode")
    st.write("3. Tokenization Vietnamese text using underthesea")
    st.write("4. Remove stop words")
    htp0 = "https://raw.githubusercontent.com/nptlong93/recommendation/main/Content_based_text_processed.png"
    st.image(htp0, caption ='Preprocessing text example' ,width=700)

    st.write("#### EDA data")
    st.write("The data is then analyzed using the Pandas Profiling library")
    htp = "https://raw.githubusercontent.com/nptlong93/recommendation/main/Content_based_EDA.png"
    st.image(htp, caption ='Duplicated product id in the dataset' ,width=700)
    st.write("=> Remove the duplicate and null values in the dataset")
    htp2= "https://raw.githubusercontent.com/nptlong93/recommendation/main/Content_based_Sub_cat.png"
    st.image(htp2, caption ='Bar chart of products in sub category' ,width=700)
    st.write("=> Balance the data by selecting 500 products from each sub category")

    st.write("#### Algorithms and Results")
    st.write("For content-based filtering, Gensim and Cosine similarity measures were used.")
    st.write("1. Gensim - a Python library for topic modeling, document indexing, and similarity retrieval with large corpora.")
    st.write("2. Cosine similarity - a mathematical measure that calculates the similarity between two non-zero vectors of an inner product space by measuring the cosine of the angle between them.")
    htp3= "https://raw.githubusercontent.com/nptlong93/recommendation/main/Content_based_result.png"
    st.image(htp3, caption ='Cosine similarity and Gensim recommendation results ' ,width=700)
    st.write("Conclusion: Both methods show a good results regarding to the mutual products and the performance. Cosine similarity was chosen for further GUI Built-in")

    st.write("#### Case 02: Collaborative Filtering")
    st.write("#### Understand the Dataset")
    st.write("The data used in this case is the user's rating of the products on Shopee.vn. The data contains 1,024,482 ratings from 650,636 users for 31,189 products. There are 4 features: product_id, user_id, user, rating.")
    st.write("The data is collected from the Shopee.vn.")
    st.write("#### Preprocessing data")
    st.write("The data is then cleaned and preprocessed using the Python library Pandas")
    htp4= "https://raw.githubusercontent.com/nptlong93/recommendation/main/Collab_rating_EDA.JPG"
    st.image(htp4, caption ='Explore the dataset' ,width=700)
    st.write("=> Processing the unbalanced data: User 199 is way more active than other users. Therefore, the data is balanced by reduced the data of user 199.")
    htp5= "https://raw.githubusercontent.com/nptlong93/recommendation/main/Collab_EDA.JPG"
    st.image(htp5, caption ='Maxtrix sparsity' ,width=700)
    st.write("=> Matrix sparsity: The matrix sparsity is 99.9%")

    st.write("#### Algorithms and Results")
    st.write("For collaborative filtering, Alternating least squares (ALS) was used.")
    htp6= "https://raw.githubusercontent.com/nptlong93/recommendation/main/RMSE_collab.JPG"
    st.image(htp6, caption ='RMSE result' ,width=700)
    st.write("Conclusion: ALS was chosen for further GUI Built-in")

    st.write("Student: Nguyen Pham Thang Long")
    st.write("This project is executed, courtesy of Mrs. Khuay Thuy Phuong")

elif choice == 'Content-based Filtering_1':
    st.subheader("Case 01: Select product to recommend")
    selected_product = st.selectbox('Select a product', data['product_name'])
    # Show product name
    st.write("### Your choice:",selected_product)
    # Show recommended products
    st.write("#### Recommended products: ")
    # Get index of product
    idx = data[data['product_name'] == selected_product].index[0]
    # Get list of similar products
    similar_products = list(enumerate(model[idx]))
    # Sort list of similar products
    sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)
    # Show random 5 similar products
    lst = [i[0] for i in sorted_similar_products[1:9]]
    # Create a seleted image
    selected_image = image_select(label= "Select image",images=[data['image'][i] for i in lst], captions=[data['product_name'][i] for i in lst], use_container_width = False)
    # Show the product info
    st.write("#### In this product:") 
    col1, col2 = st.columns(2)
    with col1:
        st.image(selected_image, width=270)
    with col2:
        st.write("[{}]({})".format(data['product_name'][data[data['image'] == selected_image].index[0]], data.iloc[idx]['link']))
        st.write("{:,} VND".format(data['price'][data[data['image'] == selected_image].index[0]]))
        st.write(star_rating(data['rating'][data[data['image'] == selected_image].index[0]]))
        #st.write("###### Description:", data['description'][data[data['image'] == selected_image].index[0]])
    # Show recommended products
    st.write("#### You might also like: ")
    # Get list of similar products based on selected image
    idx = data[data['image'] == selected_image].index[0]
    similar_products = list(enumerate(model[idx]))
    # Sort list of similar products
    sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)
    # Show random 5 similar products
    for i in sorted_similar_products[1:6]:
        # Get index of similar product
        idx = i[0]
        # Get product name   
        product_name = data.iloc[idx]['product_name']
        # Get product price
        product_price = data.iloc[idx]['price']
        # Get product rating
        product_rating = data.iloc[idx]['rating']
        # Get product image
        data[['image']] = data[['image']].astype(str)
        product_image = data.iloc[idx]['image']
        # Align 2 columns
        col1, col2 = st.columns(2)
        with col1:
            # Col 1 show product image
            st.image(product_image, width= 175 )
        with col2:
            # Show product name with big bold font and link
            st.write("[{}]({})".format(product_name, data.iloc[idx]['link']))
            # Show product price
            st.write("{:,} VND".format(product_price))
            # Show star rating
            st.write(star_rating(product_rating))

elif choice == 'Content-based Filtering_2':
    st.subheader("Case 02: Keyword to recommend")
    # Let user text input 
    text_input = st.text_input("You are looking for:")
    # Create a button
    # button = st.button('Search')
    # # If button is clicked, search for user
    # if button:
    # Preprocess text input by clean_text function
    text_input = clean_text(text_input)
    # Get list of similar products based on text input
    similar_products = find_similar_products(text_input,data['name_description_wt'])
    similar_products = get_similar_product_names(similar_products)
    # Get index list of similar products
    idx = [data[data['product_name'] == i].index[0] for i in similar_products]
    # Show 5 similar products
    for i in idx[:5]:
        # Get product name   
        product_name = data.iloc[i]['product_name']
        # Get product price
        product_price = data.iloc[i]['price']
        # Get product rating
        product_rating = data.iloc[i]['rating']
        # Get product image
        data[['image']] = data[['image']].astype(str)
        product_image = data.iloc[i]['image']
        # Align 2 columns
        col1, col2 = st.columns(2)
        with col1:
            # Col 1 show product image
            st.image(product_image, width= 175 )
        with col2:
            # Show product name with big bold font and link
            st.write("[{}]({})".format(product_name, data.iloc[i]['link']))
            # Show product price
            st.write("{:,} VND".format(product_price))
            # Show star rating
            st.write(star_rating(product_rating))
    
elif choice == 'Collaborative Filtering':
    st.subheader("Collaborative Filtering")
    # Search based on user input 
    usern = data2['user'].unique()
    def search_user(query):
        results = [user for user in usern if query.lower() in user.lower()]
        return results
    search = st.text_input('Username: ')
    textb = st.text_input('Password: ')
    # Create a button
    button = st.button('Login')
    # If button is clicked, search for user
    if button:
        search1 = search_user(search)
    # if no user is found, show error message
        if not search1:
            st.error("Your username is not correct. Please try again")
        else:
            # Show success message
            st.success("Successfully logged in")
            # Show user_id
            st.write("### Welcome,",search)
            # Show product_name based on selected user_id
            st.write("### What you might like: ")
            # Get list of product_name based on selected user_id
            product_name = data2[data2['user'] == search]['product_name'].unique()
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
                        st.image(data2.iloc[idx]['image'], width= 175)
                with col2:
                    # Show product name with big bold font and link
                    st.write("[{}]({})".format(i, data2.iloc[idx]['link']))
                    # Show product price
                    st.write("{:,} VND".format(data2.iloc[idx]['price']))
                    # Show star rating
                    st.write(star_rating(data2.iloc[idx]['rating']))

