"""WheatCare - wheat leaf disease screening demo (Streamlit).

Loads the MobileNetV2 model trained in Google Colab (wheat_disease_model.keras),
classifies an uploaded leaf photo and shows the class probabilities.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="WheatCare", page_icon="🌾", layout="centered")

HERE = Path(__file__).parent
IMG_SIZE = (224, 224)
# Same order Keras gives (alphabetical folder names) - used only if class_names.json is missing
DEFAULT_CLASSES = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]

GUIDANCE = {
    "Healthy": "No disease pattern was detected. Keep monitoring the crop regularly.",
    "BrownRust": "Brown Rust (leaf rust) usually shows as small orange-brown pustules on leaves.",
    "YellowRust": "Yellow Rust (stripe rust) usually shows as yellow stripes of pustules along the leaf.",
    "Mildew": "Powdery Mildew usually shows as white-grey powdery patches on leaves and stems.",
    "Septoria": "Septoria leaf blotch usually shows as irregular brown or tan blotches on leaves.",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Find the saved model anywhere in the repo and load it."""
    paths = sorted(HERE.rglob("*.keras"))
    if not paths:
        return None, None, None
    import tensorflow as tf
    model = tf.keras.models.load_model(paths[0])
    class_file = next(iter(HERE.rglob("class_names.json")), None)
    classes = json.loads(class_file.read_text()) if class_file else DEFAULT_CLASSES
    return model, classes, paths[0].name


def predict(model, classes, image):
    """Resize to 224x224 and return a probability per class.
    In the notebook, pixels were divided by 255 (range 0-1) before the model, so we do the same."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    return pd.Series(probs, index=classes)


st.title("🌾 WheatCare")
st.caption("Wheat leaf disease screening - AML university project (BBA AI & DS)")

if not list(HERE.rglob("*.keras")):
    st.error("Model file not found. Upload `wheat_disease_model.keras` (saved from Colab) "
             "to the top folder of the GitHub repository, then reboot the app.")
    st.write("Files this app can see in its folder:")
    st.code("\n".join(sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*")
                             if p.is_file() and ".git" not in p.parts)) or "(none)")
    st.stop()

model, classes, model_file = load_model()

st.subheader("1. Upload a wheat leaf photo")
upload = st.file_uploader("JPG or PNG image", type=["jpg", "jpeg", "png"])

st.subheader("2. Crop details (optional)")
c1, c2 = st.columns(2)
variety = c1.selectbox("Wheat variety", ["Not sure", "HD 2967", "HD 3086", "Lok 1", "GW 322"])
stage = c2.selectbox("Growth stage", ["Not sure", "Tillering", "Jointing", "Heading", "Grain filling"])
soil = c1.selectbox("Soil type", ["Not sure", "Loamy", "Clay", "Sandy"])
irrigation = c2.selectbox("Irrigation", ["Not sure", "Drip irrigation", "Flood irrigation"])
st.caption("These details are for the record only. In this version they do NOT affect the prediction "
           "(the model uses the image only).")

if st.button("Predict", type="primary"):
    if upload is None:
        st.warning("Please upload a leaf image first.")
    else:
        image = Image.open(upload)
        st.image(image, caption="Uploaded leaf", use_container_width=True)
        probs = predict(model, classes, image)
        top = probs.idxmax()
        conf = float(probs.max()) * 100

        st.metric("Predicted class", top, f"{conf:.1f}% confidence", delta_color="off")
        if conf < 60:
            st.warning("Low confidence - the photo may be unclear or the disease not covered. "
                       "Try a sharper close-up or ask an agricultural expert.")
        st.info(GUIDANCE.get(top.replace(" ", ""), "Consult an agricultural officer for guidance."))
        st.bar_chart(probs.rename("probability"))
        st.caption(f"Context: {variety} | {stage} | {soil} | {irrigation}")

with st.expander("About the model"):
    st.markdown(f"""
- **Model:** MobileNetV2 (ImageNet pre-trained, frozen) + GlobalAveragePooling, Dropout 0.3, Dense softmax
- **Input:** 224 x 224 RGB image; **Optimizer:** Adam (lr 0.001); **Loss:** sparse categorical cross-entropy
- **Dataset:** Kaggle *wheat-disease-dataset-small* - 999 images, 5 classes
  (80/20 split: 800 training, 199 validation, seed 42)
- **Classes in the loaded model:** {", ".join(classes)}
- **Model file:** `{model_file}`

**Results reported in the project report** (not recomputed by this app):

| Metric | Result |
|---|---|
| Training accuracy | 96.5% |
| Validation accuracy | 94.2% |
| Test accuracy | 92.8% |
| Precision / Recall / F1 | 92.5% / 92% / 92.2% |
""")

with st.expander("Important caveats"):
    st.markdown("""
- This is a **preliminary screening demo**, not a confirmed diagnosis. Consult an agricultural expert before any treatment.
- The model only knows the classes above; other diseases or non-wheat images will still get one of these labels.
- Trained on a small public dataset (999 images), not on Ahilyanagar field data - it is **not validated for real farm use**.
- Weather and crop details are not yet used by the model.
""")
