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
st.write("Upload multiple CT scan images to check for lung cancer classification.")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    # Preprocess images
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(image)

    # Display images
    for img in images:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize and normalize images
    img_arrays = []
    for img in images:
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_arrays.append(img_array)

    # Stack all images into a single batch (batch_size, height, width, channels)
    img_batch = np.stack(img_arrays, axis=0)

    # Predict in batch
    predictions = model.predict(img_batch)

    # Output predictions for each image
    for i, prediction in enumerate(predictions):
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        st.subheader(f"Prediction for Image {i + 1}:")
        st.success(f"**{predicted_class}** ({confidence * 100:.2f}% confidence)")
