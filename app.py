import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import zipfile
import tempfile
from keras.layers import TFSMLayer  

st.set_page_config(page_title="Jean-O-Type", layout="centered")
st.markdown("<h1 style='text-align: center;'>JEAN-O-TYPE CLASSIFIER 👖🦖</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>"
    "Upload a photo of jeans to discover your denim! "
    "Download the trained jeans-recognition model we created "
    "<a href='https://github.com/tanvividyala/jean-o-type/blob/main/jean_fit_classifier_model.zip'>here.</a>"
    "</p>", 
    unsafe_allow_html=True
)

model = None

stream = st.file_uploader('TF.Keras model file (.zip of SavedModel)', type='zip')
if stream is not None:
    with zipfile.ZipFile(stream) as myzipfile:
        with tempfile.TemporaryDirectory() as tmp_dir:
            myzipfile.extractall(tmp_dir)
            contents = myzipfile.namelist()
            root_folder = contents[0].split('/')[0]
            model_dir = os.path.join(tmp_dir, root_folder)
            
            try:
                model = TFSMLayer(model_dir, call_endpoint="serving_default")
                st.success("Model loaded using TFSMLayer (SavedModel format)")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

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

uploaded_file = st.file_uploader("Upload your jean image", type=["jpg", "jpeg", "png"])

if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(image, caption="Your uploaded image", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    raw_output = model(img_array)
    if isinstance(raw_output, dict):
        predictions = list(raw_output.values())[0].numpy()[0]
    else:
        predictions = raw_output.numpy()[0]

    top2_indices = predictions.argsort()[-2:][::-1]
    primary = jean_labels[top2_indices[0]]
    secondary = jean_labels[top2_indices[1]]
    confidence = predictions[top2_indices[0]] * 100
    second_conf = predictions[top2_indices[1]] * 100

    st.markdown(f"### 🎉 You got: **{primary}** ({confidence:.2f}% confidence)")
    st.markdown(f"👖 Description: *{jean_descriptions[primary]['desc']}*")
    st.markdown(f"🕰️ Popular in: **{jean_descriptions[primary]['trending years']}**")

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(f"dino_pics/{jean_descriptions[primary]['dino_img']}", caption="Dino rocking the style!", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"💡 Second guess: **{secondary}** ({second_conf:.2f}%)")
    st.markdown(f"*{jean_descriptions[secondary]['desc']}* — **{jean_descriptions[secondary]['trending years']}**")
elif uploaded_file and model is None:
    st.warning("Please upload a model file before analyzing an image.")