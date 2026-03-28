
🧠 MediScan AI

AI-powered Medical Image Analysis System for Chest X-rays and CT Scans

🚀 Overview

MediScan AI is a deep learning-based system that analyzes chest medical images (X-ray and CT scans) to:

Identify image type (X-ray or CT)
Detect diseases such as Pneumonia and Tuberculosis
Highlight important regions using Grad-CAM visualization

This tool is designed to assist in early diagnosis and provide visual explanations of AI predictions.

🎯 Problem Statement

Manual analysis of medical images is time-consuming and requires expert knowledge. In many regions, access to radiologists is limited.

💡 Solution

MediScan AI automates:

Medical image classification
Disease detection
Visual explanation using heatmaps
⚙️ Features
🖼️ Image Type Detection (X-ray / CT)
🦠 Disease Classification (Normal / Pneumonia / TB)
🔥 Grad-CAM Heatmap Visualization
📊 Confidence Score Display
🌐 Interactive Web Interface (Streamlit)
🛠️ Tech Stack
Python
TensorFlow / Keras
MobileNetV2 (Transfer Learning)
OpenCV
Streamlit
📂 Project Structure
chest_xray/
│
├── app.py
├── model/
│   ├── type_model.h5
│   └── disease_model.h5
├── utils/
│   └── gradcam.py
├── dataset/
└── README.md
▶️ How to Run
1. Clone Repository
git clone <your-repo-link>
cd chest_xray
2. Install Dependencies
pip install -r requirements.txt
3. Run Application
streamlit run app.py


🧠 How It Works
User uploads an image
Model predicts image type
Disease detection model analyzes the image
Grad-CAM highlights important regions
Results are displayed with confidence score
