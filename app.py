import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.preprocessing import image

# Load the trained model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Load Validation Dataset (For Class Names)
base_dir = "./archive/New Plant Diseases Dataset(Augmented)/"
valid_dir = f"{base_dir}/New Plant Diseases Dataset(Augmented)/valid"
validation_set = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

class_names = validation_set.class_names

# Streamlit app title
st.title("Plant Disease Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert byte data to image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = cnn.predict(img_array)
    predicted_label = class_names[np.argmax(predictions[0])]

    # Display the image and prediction
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted: {predicted_label}")