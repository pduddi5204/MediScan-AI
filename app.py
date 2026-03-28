import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from utils.gradcam import get_gradcam
import cv2
import pandas as pd

# load models
type_model = load_model("model/type_model.h5")
disease_model = load_model("model/disease_model.h5")

# UI
st.set_page_config(page_title="MediScan AI", layout="centered")

st.title("🧠 MediScan AI")
st.subheader("AI-powered Chest Scan Analysis")

# 🔥 Sidebar Mode
mode = st.sidebar.selectbox("Select Mode", ["User", "Doctor"])

# ===================== USER MODE =====================
if mode == "User":

    st.markdown("### 👤 Patient Details")

    name = st.text_input("Name")
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("### 📤 Upload Medical Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # IMAGE PREPROCESS
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

        # RESULT UI
        st.success(f"Image Type: {img_type}")
        st.info(f"Disease: {disease}")

        confidence = float(np.max(disease_pred)*100)
        st.progress(confidence/100)
        st.write(f"Confidence: {confidence:.2f}%")

        # 🔥 GRAD-CAM (simple)
        heatmap = get_gradcam(disease_model, img)
        heatmap = cv2.resize(heatmap, (224,224))

        original = img[0]

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + original * 255

        st.image(superimposed_img.astype("uint8"), caption="Highlighted Area 🔥")

        st.write("🔍 Highlighted area shows the region influencing AI prediction")

        # 🔥 SAVE DATA
        data = {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Type": img_type,
            "Disease": disease,
            "Confidence": confidence
        }

        df = pd.DataFrame([data])

        try:
            old = pd.read_csv("records.csv")
            df = pd.concat([old, df])
        except:
            pass

        df.to_csv("records.csv", index=False)

        st.success("✅ Record Saved Successfully!")

# ===================== DOCTOR MODE =====================
elif mode == "Doctor":

    st.markdown("### 👨‍⚕️ Patient Records")

    try:
        df = pd.read_csv("records.csv")

        st.dataframe(df)

        # 🔥 Simple filter
        disease_filter = st.selectbox("Filter by Disease", ["All"] + list(df["Disease"].unique()))

        if disease_filter != "All":
            df = df[df["Disease"] == disease_filter]
            st.write("Filtered Results:")
            st.dataframe(df)

    except:
        st.warning("No records found yet.")
