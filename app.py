import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model("D://Projects//Python//Mushroom Vision//Capstone-Project---Mushroom-Vision//model.h5")  

def predict_mushroom(img):
    img_array = np.array(img.resize((150, 150))) / 255.0  # Resize and normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
 
    with open('class_names.txt', 'r') as f:
     class_names = [line.strip() for line in f]
    return class_names[predicted_class[0]]


st.title("Mushroom Type Predictor")
st.write("Upload an image of a mushroom, and this app will predict its type!")

uploaded_file = st.file_uploader("Choose a mushroom image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Mushroom Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    label = predict_mushroom(image)
    st.write(f"The mushroom type is: {label}")
