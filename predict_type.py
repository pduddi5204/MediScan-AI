from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# model load
model = load_model("model/type_model.h5")

# image load
img = image.load_img("test.jpeg", target_size=(224,224))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# prediction
pred = model.predict(img)

print("Raw prediction:", pred)

classes = ["CT-Scan", "X-RAY" ]

print("Prediction:", classes[np.argmax(pred)])
confidence = np.max(pred) * 100
print("Confidence:", confidence, "%")