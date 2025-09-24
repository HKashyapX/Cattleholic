import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Cattleholic: Breed Recognition")

# --- Load the Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('cattleholic_model.h5')
    return model

model = load_my_model()

# --- Prediction Function ---
def predict(image_to_predict):
    # Preprocess the image
    img = image_to_predict.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get the class names from the training data (must match the order from training)
    # This is a bit of a hack for the hackathon; in a real app, save class names with the model.
    import os
    class_names = sorted(os.listdir('data')) # Assuming 'data' directory exists and subdirs are class names

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


# --- Web App Interface ---
st.title("🐮 Cattleholic 🐄")
st.write("AI-powered breed recognition for Indian Cattle and Buffaloes.")
st.write("Upload an image and the AI will predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        st.write("")
        st.write("### Predicting...")
        
        # Make a prediction
        predicted_breed, confidence_score = predict(image)

        st.success(f"**Predicted Breed:** {predicted_breed.replace('_', ' ').title()}")
        st.info(f"**Confidence:** {confidence_score:.2f}%")