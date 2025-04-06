import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import zipfile
import tempfile

st.set_page_config(page_title="👖🦖", layout="centered")
st.markdown("<h1 style='text-align: center;'>JEAN-O-TYPE CLASSIFIER 👖🦖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a photo of jeans to discover your denim destiny!</p>", unsafe_allow_html=True)

# 👇 Choose model input method
upload_type = st.radio("Choose model format:", ["Zip (.zip)", "HDF5 (.h5)"])

model = None  # Initialize

if upload_type == "Zip (.zip)":
    stream = st.file_uploader('Upload model as zip (containing .h5 file)', type='zip')
    if stream is not None:
        with zipfile.ZipFile(stream) as myzipfile:
            with tempfile.TemporaryDirectory() as tmp_dir:
                myzipfile.extractall(tmp_dir)
                # Get first file inside (assumes it’s the model file)
                inner_files = myzipfile.namelist()
                model_file = [f for f in inner_files if f.endswith(".h5") or f.endswith(".keras")]
                if model_file:
                    model_path = os.path.join(tmp_dir, model_file[0])
                    model = tf.keras.models.load_model(model_path)
                else:
                    st.error("No .h5 or .keras file found inside the zip.")

elif upload_type == "HDF5 (.h5)":
    stream = st.file_uploader('Upload .h5 model file', type='h5')
    if stream is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file.write(stream.read())
            model = tf.keras.models.load_model(tmp_file.name)

# Jean label info
jean_labels = ['Baggy', 'Bootcut', 'Skinny', 'Straight', 'Wide-Leg']

jean_descriptions = {
    'Skinny': {
        'desc': "Tight-fitting and sleek, perfect for the 2010s aesthetic.",
        'trending years': "2005–2018",
        'dino_img': "dino_skinny.png"
    },
    'Bootcut': {
        'desc': "A 90s classic with a flare at the bottom for boots.",
        'trending years': "1995–2005",
        'dino_img': "dino_bootcut.png"
    },
    'Baggy': {
        'desc': "Roomy and rebellious, born from 90s skater culture.",
        'trending years': "1990s–early 2000s, resurged in 2020s",
        'dino_img': "dino_baggy.png"
    },
    'Wide-Leg': {
        'desc': "Free-flowing and bold — a favorite for comfort and flair.",
        'trending years': "1970s, resurged post-2018",
        'dino_img': "dino_wide.png"
    },
    'Straight': {
        'desc': "Simple, classic, and effortlessly cool — straight through the leg.",
        'trending years': "Forever in style",
        'dino_img': "dino_straight.png"
    }
}

# Image input (only once model is loaded)
if model is not None:
    uploaded_file = st.file_uploader("Upload your jean image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded image", width=300)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        top2_indices = predictions.argsort()[-2:][::-1]

        primary = jean_labels[top2_indices[0]]
        secondary = jean_labels[top2_indices[1]]
        confidence = predictions[top2_indices[0]] * 100
        second_conf = predictions[top2_indices[1]] * 100

        # Output results
        st.markdown(f"### 🎉 You got: **{primary}** ({confidence:.2f}% confidence)")
        st.markdown(f"👖 Description: *{jean_descriptions[primary]['desc']}*")
        st.markdown(f"🕰️ Popular in: **{jean_descriptions[primary]['trending years']}**")
        st.image(f"dino_pics/{jean_descriptions[primary]['dino_img']}", caption="Dino rocking the style!", width=300)

        # Show runner-up
        st.markdown("---")
        st.markdown(f"💡 Second guess: **{secondary}** ({second_conf:.2f}%)")
        st.markdown(f"*{jean_descriptions[secondary]['desc']}* — **{jean_descriptions[secondary]['trending years']}**")
else:
    st.info("Please upload a model file to continue.")
