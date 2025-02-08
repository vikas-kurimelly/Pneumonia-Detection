# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_model.keras')

model = load_model()

def preprocess_image(image):
    # Resize the image to match the input size of your model
    img = image.resize((64, 64))  # Adjusted to match the model's input size
    img_array = np.array(img)
    # Convert to grayscale if the image is not already
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    # Normalize the image
    img_array = img_array / 255.0
    # Add channel and batch dimensions
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pneumonia(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

# Streamlit UI
st.title('Pneumonia Detection from X-ray Images')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    st.write("")
    
    if st.button('Classify Image'):
        st.write("Classifying...")
        
        prediction = predict_pneumonia(image)
        
        if prediction > 0.6:
            st.error(f"Pneumonia detected with {prediction:.2%} confidence")
        else:
            st.success(f"No pneumonia detected. Normal with {1-prediction:.2%} confidence")

st.write("Note: This app is for demonstration purposes only. Always consult with a medical professional for accurate diagnosis.")
