
import json, numpy as np, pandas as pd, streamlit as st, tensorflow as tf
from PIL import Image

st.set_page_config(page_title="WheatCare", page_icon="🌾")
st.title("🌾 WheatCare")
st.caption("Wheat leaf disease screening demo - Ahilyanagar district, Maharashtra")
st.warning("Preliminary screening only, not a confirmed diagnosis. Consult an agricultural expert.")

@st.cache_resource
def load():
    return tf.keras.models.load_model("wheatcare_model.keras"), json.load(open("class_names.json"))

model, classes = load()

st.subheader("Crop details")
c1, c2 = st.columns(2)
variety = c1.selectbox("Wheat type", ["Bread wheat", "Durum (macaroni) wheat", "Khapli (emmer) wheat", "Sharbati", "Lokwan", "Other / unknown"])
stage = c2.selectbox("Growth stage", ["Tillering", "Jointing", "Heading", "Flowering", "Grain filling"])
soil = c1.selectbox("Soil type", ["Black cotton", "Loamy", "Clay", "Sandy", "Red", "Other"])
irrigation = c2.selectbox("Irrigation", ["Drip", "Flood", "Sprinkler", "Rain-fed"])
t1, t2, t3 = st.columns(3)
temp = t1.number_input("Temperature (°C)", -5.0, 50.0, 20.0)
hum = t2.number_input("Humidity (%)", 0.0, 100.0, 60.0)
rain = t3.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)

st.subheader("Leaf image")
f = st.file_uploader("Upload a wheat leaf image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    st.image(img, width=300)
    if st.button("Detect disease"):
        x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
        p = model.predict(x, verbose=0)[0]
        i = int(p.argmax())
        st.success(f"Prediction: {classes[i]}  |  Confidence: {p[i]:.1%}")
        st.bar_chart(pd.Series({classes[j]: float(p[j]) for j in range(len(classes))}))
        if p[i] < 0.6:
            st.info("Low confidence - please upload a clearer photo or consult an expert.")
        st.caption(f"Crop details noted: {variety}, {stage}, {soil} soil, {irrigation} irrigation, {temp}°C, {hum}% humidity, {rain} mm rain. These details are recorded for context and do not change the image prediction.")
