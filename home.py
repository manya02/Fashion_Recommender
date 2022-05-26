# import email
# from logging import exception
# from re import A
# from isort import file
# from more_itertools import first
# from sqlalchemy import column
import streamlit as st
from PIL import Image


import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle


import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

features_list = np.array(pickle.load(open('feature_details.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


st.title('Fashion Showcase')
st.text("##               An Enterance to Fashionable World                   ##")
nav = st.sidebar.radio("Navigation :",["Home","Products","Order Now","Contact Us"])

# file upload
uploaded_file= st.file_uploader("Choose an Image")



# load file(features extraction)
model = ResNet50(weights = 'imagenet',include_top=False, input_shape= (224,224,3))
model.trianable=False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def saveUploadedFile(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1 
    except:
        return 0


def extract_features(img_path, model):
    
    img = image.load_img(img_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normal_result = result/norm(result)
    return normal_result
    
# generate recommendation
def recommend(features,features_list):
    neighbors = NearestNeighbors(n_neighbors = 6, algorithm = 'brute', metric = 'euclidean')
    neighbors.fit(features_list)
    distences , indices = neighbors.kneighbors([features])
    return indices


def generate_img (details_file, names_file,n,features):
    features_list = pickle.load(open(details_file,'rb'))
    filenames = pickle.load(open(names_file ,'rb'))

    neighbors = NearestNeighbors(n_neighbors = n , algorithm = 'brute', metric = 'euclidean')
    neighbors.fit(features_list)
    

    distances , indices = neighbors.kneighbors([features])
    return indices,filenames
    

# show
if uploaded_file is not None:
    if saveUploadedFile(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.text("your sample image")
        st.image(display_image)
        features = extract_features(os.path.join('uploads',uploaded_file.name),model)
        st.text("**                   Our recommendations                   **")
        indices = recommend(features,features_list)
        col1,col2,col3,col4,col5,col6= st.columns(6)
        with col1:
            st.image(filenames[indices[0][1]])
            st.text("Img 1")
        with col2:
            st.image(filenames[indices[0][2]])
            st.text("Img 2")
        with col3:
            st.image(filenames[indices[0][3]])
            st.text("Img 3")
        with col4:
            st.image(filenames[indices[0][4]])
            st.text("Img 4")
        with col5:
            st.image(filenames[indices[0][5]])
            st.text("Img 5")
        
        st.text("\n\n")
        option = st.selectbox(
        "Hey!!!   Please choose the type of given sample so that we can generate complete recommendation for our special product (Women's Top And Men's Shirt ) which includes (shoes,sandals,jeans,watches,spectacles)",
        ('Others Item', "Men's Shirt", "Women's Top"))
        st.text("\n\n")
        st.write('Complete recommendation for :', option)
        if option == "Women's Top":
            indices,filenames = generate_img('feature_details_jeans.pkl','filenames_jeans.pkl',1,features)
            st.text("Jeans")
            st.image(filenames[indices[0][0]])
            indices,filenames = generate_img('feature_details_sandals.pkl','filenames_sandals.pkl',1,features)
            st.text("Shoes/Sandals")
            st.image(filenames[indices[0][0]])
            indices,filenames = generate_img('feature_details_spec.pkl','filenames_spec.pkl',1,features)
            st.text("Goggles")
            st.image(filenames[indices[0][0]])
        elif option == "Men's Shirt":
            indices,filenames = generate_img('feature_details_mensjeans.pkl','filenames_mensjeans.pkl',1,features)
            st.text("jeans")
            st.image(filenames[indices[0][0]])
            indices,filenames = generate_img('feature_details_shoe.pkl','filenames_shoe.pkl',1,features)
            st.text("Shoes/Sandals")
            st.image(filenames[indices[0][0]])
            indices,filenames = generate_img('feature_details_spec.pkl','filenames_spec.pkl',1,features)
            st.text("Goggles")
            st.image(filenames[indices[0][0]])
        

                
    else:
        st.header("Some error occured during file uploadation")

    
    st.title("Order Now") 
    first,last = st.columns(2)

    first.text_input("First Name")
    last.text_input("Last Name")

    email,mob = st.columns([3,1])
    email.text_input("Email Id")
    mob.text_input("mobile num")
    
    # third = st.columns(1)
    # third.text_input("Shipping Address")

    ch,b1,sub = st.columns(3)
    ch.checkbox("I Agree")
    sub.button("Submit")








    










