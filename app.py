import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


# Load pneumonia detection model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_model.keras')

model = load_model()

def is_black_and_white(image):
    """Checks if the uploaded image is grayscale (black & white)."""
    img_array = np.array(image)

    # If image has 3 channels (RGB), check if all channels are almost identical
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        if np.allclose(r, g, atol=5) and np.allclose(g, b, atol=5):
            return True
        return False

    # If the image has only one channel, it's already grayscale
    return True

def preprocess_image(image):
    """Preprocess the X-ray image for pneumonia detection."""
    image = image.resize((64, 64))  # Resize to model input size
    img_array = np.array(image) / 255.0  # Normalize
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=-1)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_pneumonia(image):
    """Runs the pneumonia classification model."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    return prediction

# Streamlit UI
st.title("🩺 Pneumonia Detection")
st.write("Upload a black-and-white (grayscale) chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded X-ray Image", use_column_width=True)

    if st.button("🔍 Classify Image"):
        with st.spinner("Checking if this is a black & white X-ray..."):
            if not is_black_and_white(image):
                st.error("🚫 This image is not black and white! Please upload a grayscale chest X-ray.")
            else:
                with st.spinner("Analyzing for pneumonia..."):
                    prediction = predict_pneumonia(image)
                    if prediction > 0.5:
                        st.error(f"⚠️ **Pneumonia detected** with **{prediction:.2%} confidence**.")
                    else:
                        st.success(f"✅ **No pneumonia detected**. Normal with **{(1 - prediction):.2%} confidence**.")

# Disclaimer
st.write("⚠️ **Note:** This app is for demonstration purposes only. Always consult a medical professional for an accurate diagnosis.")
