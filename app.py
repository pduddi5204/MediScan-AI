import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from utils.gradcam import get_gradcam
import cv2
import pandas as pd
import time

# ---------------- SESSION ----------------
if "started" not in st.session_state:
    st.session_state.started = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MediScan AI", layout="centered")

# ---------------- STYLING ----------------
st.markdown("""
<style>
.card {
    padding:20px;
    border-radius:15px;
    background:linear-gradient(to right,#1e3c72,#2a5298);
    color:white;
    box-shadow:0px 5px 15px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>🧠 MediScan AI</h1>
<p style='text-align:center;'>Smart AI Medical Diagnosis System</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- MODELS ----------------
type_model = load_model("model/type_model.h5")
disease_model = load_model("model/disease_model.h5")

# ---------------- SIDEBAR ----------------
mode = st.sidebar.selectbox("Select Mode", ["User", "Doctor"])

# ================= USER =================
if mode == "User":

    # HERO
    st.markdown("""
    <div style="text-align:center;padding:30px;background:#1e3c72;color:white;border-radius:15px;">
    <h2>MediScan AI</h2>
    <p>AI-powered Chest Scan Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    # FEATURES
    st.markdown("""
    ### ✨ Features
    - AI Detection  
    - Heatmap Visualization  
    - Report Generation  
    - Doctor Dashboard  
    """)

    # START BUTTON
    if st.button("🚀 Start Scan"):
        st.session_state.started = True

    # FORM
    if st.session_state.started:

        st.markdown("### 👤 Patient Details")

        name = st.text_input("Name")
        age = st.number_input("Age", 1, 100)
        gender = st.selectbox("Gender", ["Male","Female","Other"])

        st.markdown("### 📤 Upload Scan")
        uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

        if uploaded_file:

            st.image(uploaded_file)

            img = image.load_img(uploaded_file, target_size=(224,224))
            img = image.img_to_array(img)/255.0
            img = np.expand_dims(img, axis=0)

            # SCAN ANIMATION
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            # PREDICT
            type_pred = type_model.predict(img)
            disease_pred = disease_model.predict(img)

            img_type = ["CT","XRAY"][np.argmax(type_pred)]
            disease = ["NORMAL","PNEUMONIA","TB"][np.argmax(disease_pred)]
            confidence = float(np.max(disease_pred)*100)

            # RESULT CARD
            st.markdown(f"""
            <div class="card">
            <h3>🧠 AI Diagnosis</h3>
            <p>Type: {img_type}</p>
            <p>Disease: {disease}</p>
            <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # REPORT
            st.markdown(f"""
            ### 🧾 Report
            Name: {name}  
            Age: {age}  
            Disease: {disease}  
            Confidence: {confidence:.2f}%
            """)

            # SAVE
            data = {
                "Name": name,
                "Age": age,
                "Gender": gender,
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

            # HEATMAP
            heatmap = get_gradcam(disease_model, img)
            heatmap = cv2.resize(heatmap, (224,224))

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            original = np.uint8(255 * img[0])
            result = heatmap * 0.4 + original

            st.image(result.astype("uint8"), caption="🔥 Highlighted Area")

# ================= DOCTOR =================
elif mode == "Doctor":

    st.session_state.started = False  # 🔥 important

    st.markdown("### 👨‍⚕️ Doctor Dashboard")

    try:
        df = pd.read_csv("records.csv")

        if df.empty:
            st.warning("No records")
        else:
            st.dataframe(df)
            st.bar_chart(df["Disease"].value_counts())

        if st.button("🗑️ Delete All"):
            pd.DataFrame().to_csv("records.csv", index=False)
            st.rerun()

    except:
        st.warning("No data yet")
