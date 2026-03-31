import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from utils.gradcam import get_gradcam
import cv2
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ---------------- FILE SAFETY ----------------
if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["ID","Password","Role"]).to_csv("users.csv", index=False)

if not os.path.exists("records.csv"):
    pd.DataFrame(columns=[
        "Patient","Age","Gender","Doctor","Centre",
        "Type","Disease","Confidence","Image","Heatmap","PDF","Status"
    ]).to_csv("records.csv", index=False)

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Medical System", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<div style="background:#1e3c72;padding:15px;color:white;text-align:center;font-size:22px;">
🏥 Medical Report Management System
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as: {st.session_state.user}")
    role = st.session_state.role
else:
    role = st.sidebar.selectbox(
        "Select Option",
        ["Login", "Create Account"]
    )

if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------- LOAD MODEL ----------------
type_model = load_model("model/type_model.h5", compile=False)
disease_model = load_model("model/disease_model.h5", compile=False)

# ---------------- PDF FUNCTION ----------------
def generate_pdf(data):
    file_name = f"report_{data['Patient']}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Medical Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Patient: {data['Patient']}", styles['Normal']))
    elements.append(Paragraph(f"Age: {data['Age']}", styles['Normal']))
    elements.append(Paragraph(f"Gender: {data['Gender']}", styles['Normal']))
    elements.append(Paragraph(f"Disease: {data['Disease']}", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {data['Confidence']}", styles['Normal']))

    elements.append(Spacer(1, 10))

    if os.path.exists(data["Image"]):
        elements.append(RLImage(data["Image"], width=250, height=250))

    if os.path.exists(data["Heatmap"]):
        elements.append(RLImage(data["Heatmap"], width=250, height=250))

    doc.build(elements)
    return file_name

# ================= CREATE ACCOUNT =================
if role == "Create Account":

    st.subheader("Create Account")

    account_type = st.selectbox("Account Type", ["Doctor", "Diagnosis Centre"])
    user_id = st.text_input("ID")
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):
        df = pd.read_csv("users.csv")

        if user_id in df["ID"].values:
            st.error("User already exists")
        else:
            df = pd.concat([df, pd.DataFrame([{
                "ID": user_id,
                "Password": password,
                "Role": account_type
            }])])
            df.to_csv("users.csv", index=False)
            st.success("Account Created")

# ================= LOGIN =================
elif role == "Login":

    st.subheader("Login")

    login_id = st.text_input("ID")
    login_pass = st.text_input("Password", type="password")

    if st.button("Login"):
        df = pd.read_csv("users.csv")
        user = df[(df["ID"] == login_id) & (df["Password"] == login_pass)]

        if not user.empty:
            st.session_state.logged_in = True
            st.session_state.user = login_id
            st.session_state.role = user.iloc[0]["Role"]
            st.rerun()
        else:
            st.error("Invalid credentials")

# ================= CENTRE =================
if role == "Diagnosis Centre":


    st.subheader("Upload Report")

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    uploaded_file = st.file_uploader("Upload X-ray")
    doctor_id = st.text_input("Doctor ID")

    if uploaded_file:
        st.image(uploaded_file)

        img = image.load_img(uploaded_file, target_size=(224,224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        type_pred = type_model.predict(img)
        disease_pred = disease_model.predict(img)

        img_type = ["CT","XRAY"][np.argmax(type_pred)]
        disease = ["NORMAL","PNEUMONIA","TB"][np.argmax(disease_pred)]
        confidence = float(np.max(disease_pred)*100)

        st.success(f"{disease} ({confidence:.2f}%)")

        # HEATMAP
        heatmap = get_gradcam(disease_model, img)
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = np.uint8(255 * img[0])
        result = heatmap * 0.4 + original

        st.image(result.astype("uint8"))

        if st.button("Submit"):

            if not name or not doctor_id:
                st.error("Fill all fields")
                st.stop()

            os.makedirs("uploads", exist_ok=True)

            img_path = f"uploads/{uploaded_file.name}"
            heatmap_path = f"uploads/heat_{uploaded_file.name}"

            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            cv2.imwrite(heatmap_path, result)

            data = {
                "Patient": name,
                "Age": age,
                "Gender": gender,
                "Doctor": doctor_id,
                "Centre": st.session_state.user,
                "Type": img_type,
                "Disease": disease,
                "Confidence": confidence,
                "Image": img_path,
                "Heatmap": heatmap_path,
                "Status": "Pending"
            }

            pdf_file = generate_pdf(data)
            data["PDF"] = pdf_file

            df = pd.read_csv("records.csv")
            df = pd.concat([df, pd.DataFrame([data])])
            df.to_csv("records.csv", index=False)

            st.success("Report sent successfully")

# ================= DOCTOR =================
if role == "Doctor":

    st.subheader("Doctor Dashboard")

    df = pd.read_csv("records.csv")
    df = df[df["Doctor"] == st.session_state.user]

    if df.empty:
        st.warning("No reports")
    else:
        for i, row in df.iterrows():

            st.markdown("---")
            st.write("Patient:", row["Patient"])
            st.write("Disease:", row["Disease"])
            st.write("Confidence:", row["Confidence"])
            st.write("Status:", row["Status"])

            st.image(row["Image"])
            st.image(row["Heatmap"])

            col1, col2 = st.columns(2)

            if col1.button(f"Approve {i}"):
                df.loc[i, "Status"] = "Approved"
                df.to_csv("records.csv", index=False)
                st.rerun()

            if col2.button(f"Reject {i}"):
                df.loc[i, "Status"] = "Rejected"
                df.to_csv("records.csv", index=False)
                st.rerun()

            if os.path.exists(row["PDF"]):
                with open(row["PDF"], "rb") as f:
                    st.download_button(f"Download PDF {i}", f, file_name=row["PDF"])

# ---------------- FOOTER ----------------
st.markdown("<hr><center>Medical System © 2026</center>", unsafe_allow_html=True)