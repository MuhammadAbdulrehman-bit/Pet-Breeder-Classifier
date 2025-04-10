import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np


model = tf.keras.models.load_model("pet_breed_classifier.keras")  #Load Model

# Label names (from Oxford-IIIT Pet dataset)
class_names = [
    'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer',
    'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher',
    'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue',
    'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
    'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier',
    'yorkshire_terrier'
]

# UI
st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐾", layout="centered")
st.title("🐶🐱 Pet Breed Classifier")
st.markdown("Upload an image of a **cat or dog**, and let the AI tell you its **breed**!")

uploaded_file = st.file_uploader("📁 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img = np.array(img).astype("float32")
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0) 

        # Predict
        preds = model.predict(img)
        predicted_index = np.argmax(preds)
        predicted_class = class_names[predicted_index]
        confidence = 100 * np.max(preds)

        st.success(f"🎯 Predicted Breed: **{predicted_class.capitalize()}**")
        st.info(f"Confidence: `{confidence:.2f}%`")

        # Optional: Top 3 breeds
        st.subheader("🔍 Top 3 Predictions")
        top_3 = np.argsort(preds[0])[-3:][::-1]
        for i in top_3:
            st.write(f"• {class_names[i].capitalize()} ({preds[0][i]*100:.2f}%)")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
