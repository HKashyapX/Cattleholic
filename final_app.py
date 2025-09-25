import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# A simple "database" of breed features for all 20 breeds
BREED_FEATURES = {
    'amrit_mahal': [
        'Known for its long, sharp horns that emerge close together and curve backward.',
        'A majestic draught breed from Karnataka, often grey in color.',
        'Has a distinctive, well-defined hump.'
    ],
    'bachaur': [
        'A compact, medium-sized draught animal.',
        'Known for its good walking and running ability.',
        'Horns are medium-sized and curve upwards and inwards.'
    ],
    'bargur': [
        'A hill cattle breed, known for being aggressive and difficult to train.',
        'Typically brown with white markings.',
        'Compact and well-suited for trotting in hilly terrain.'
    ],
    'deoni': [
        'Easily identified by its distinctive black and white spotted patches.',
        'Ears are long and drooping.',
        'A dual-purpose animal, good for both milk and work.'
    ],
    'gaolao': [
        'A light-bodied animal, often white or grey in color.',
        'Face is notably long and narrow.',
        'Horns are short and stubby, pointing backwards.'
    ],
    'gir': [
        'Has a prominent, rounded, and convex forehead.',
        'Ears are extremely long, pendulous, and folded like a leaf.',
        'Horns are curved, starting at the base of the head and growing back and upwards.'
    ],
    'hallikar': [
        'Famous for its very long, sharp, and backward-curving horns.',
        'A strong and compact draught breed from Karnataka.',
        'The horns emerge vertically from the top of the head.'
    ],
    'kangayam': [
        'A strong draught breed, often white or grey in color.',
        'Horns curve outwards and then inwards, forming a crescent shape.',
        'Has dark markings around the eyes and on the knees.'
    ],
    'kankrej': [
        'Has large, lyre-shaped horns that curve outwards and upwards.',
        'Known for its "gait," a unique 1.25x speed walking style.',
        'Face is short with large, expressive eyes.'
    ],
    'khillari': [
        'A powerful draught breed known for its speed and endurance.',
        'Horns are long and pointed, sweeping back and then upwards in a distinctive curve.',
        'Often greyish-white in color.'
    ],
    'krishna_valley': [
        'A large, heavy breed with a powerful build.',
        'Horns are small and curve upwards and inwards.',
        'Often has a loose, pendulous dewlap (skin under the neck).'
    ],
    'malnad_gidda': [
        'A dwarf (gidda) breed from the Malnad region of Karnataka.',
        'Very small and compact, often black or brown.',
        'Well-adapted to heavy rainfall and hilly terrain.'
    ],
    'mewati': [
        'A strong dual-purpose breed, good for work and milk.',
        'Horns emerge from the side of the head and curve upwards.',
        'Generally docile and calm in temperament.'
    ],
    'nagori': [
        'A large, white, and muscular draught breed from Rajasthan.',
        'Known for its trotting ability and speed.',
        'The back is straight and powerful.'
    ],
    'ongole': [
        'A very large, muscular breed, often pure white.',
        'Known for its strength, stamina, and disease resistance.',
        'Has a large, fleshy dewlap and a prominent hump.'
    ],

    'rathi': [
        'A good dairy breed, often brown or black with white patches.',
        'Known for its docile nature and high milk yield.',
        'Originated from the arid regions of Rajasthan.'
    ],
    'red_sindhi': [
        'Known for its deep, reddish-brown color.',
        'A popular, high-yielding dairy breed.',
        'Horns are thick, short, and curve outwards.'
    ],
    'sahiwal': [
        'One of the best dairy breeds of Zebu cattle.',
        'Has a very fleshy, pendulous dewlap and a prominent hump.',
        'Typically reddish-dun or pale red in color.'
    ],
    'tharparkar': [
        'A hardy dual-purpose breed from the Thar Desert.',
        'Almost always white or light grey in color.',
        'Can thrive in very harsh, arid conditions.'
    ],

    'vechur': [
        'Officially the smallest cattle breed in the world (a dwarf breed).',
        'Extremely compact, with a light brown or black coat.',
        'Known for producing milk with a high fat content.'
    ]
}
def get_features(breed_name):
    """Looks up the features for a given breed name."""
    # Clean up the name to match the dictionary keys (lowercase, no spaces)
    clean_name = breed_name.lower().replace(' ', '_')
    return BREED_FEATURES.get(clean_name, []) # Returns the list of features or an empty list if not found
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

        # --- THIS IS THE NEW PART THAT DISPLAYS THE FEATURES ---
        features = get_features(predicted_breed)
        if features:
            st.write("---") # Adds a nice separator line
            st.write("#### Distinctive Features:")
            for feature in features:
                st.markdown(f"- {feature}")