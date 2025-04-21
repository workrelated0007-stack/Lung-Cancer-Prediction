import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (replace with your actual model file)
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ['Benign', 'Malignant', 'Normal']

# Set expected input image size (replace with actual model input shape)
IMG_SIZE = (256, 256)

# Title
st.title("Lung Cancer Detection from CT Scan")
st.write("Upload a CT scan image to check for lung cancer classification.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.subheader("Prediction:")
    st.success(f"**{predicted_class}** ({confidence*100:.2f}% confidence)")
