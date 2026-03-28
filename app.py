import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from utils.gradcam import get_gradcam
import cv2

# load models
type_model = load_model("model/type_model.h5")
disease_model = load_model("model/disease_model.h5")

st.title("🧠 Medical AI Detector")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = image.load_img(uploaded_file, target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # TYPE
    type_pred = type_model.predict(img)
    type_classes = ["CT","XRAY"]
    img_type = type_classes[np.argmax(type_pred)]

    # DISEASE
    disease_pred = disease_model.predict(img)
    disease_classes = ["NORMAL","PNEUMONIA","TB"]
    disease = disease_classes[np.argmax(disease_pred)]

    st.write("### Image Type:", img_type)
    st.write("### Disease:", disease)
    st.write("Confidence:", np.max(disease_pred)*100, "%")

    heatmap = get_gradcam(disease_model, img)
    heatmap = cv2.resize(heatmap, (224,224))
    original = img[0]

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + original * 255

    st.image(superimposed_img.astype("uint8"), caption="Highlighted Area ")