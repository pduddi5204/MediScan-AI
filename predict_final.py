from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# load models
type_model = load_model("model/type_model.h5")
disease_model = load_model("model/disease_model.h5")

# load image
img = image.load_img("test.png", target_size=(224,224))
img = image.img_to_array(img)
img = img / 255.0

# grayscale fix
if len(img.shape) == 2 or img.shape[-1] == 1:
    img = np.stack((img,)*3, axis=-1)

img = np.expand_dims(img, axis=0)

# =========================
# STEP 1: TYPE DETECTION
# =========================
type_pred = type_model.predict(img)
type_classes = ["CT", "XRAY"]

confidence_type = np.max(type_pred)

# 🔥 reject logic
if confidence_type < 0.7:
    print("❌ Not a valid medical image")
else:
    img_type = type_classes[np.argmax(type_pred)]
    print("Image Type:", img_type)
    print("Type Confidence:", confidence_type * 100, "%")

    # =========================
    # STEP 2: DISEASE DETECTION
    # =========================
    disease_pred = disease_model.predict(img)

    disease_classes = ["NORMAL", "PNEUMONIA", "Tuberculosis"]
    disease = disease_classes[np.argmax(disease_pred)]

    print("Disease:", disease)
    print("Disease Confidence:", np.max(disease_pred) * 100, "%")