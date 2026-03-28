from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset_disease/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    'dataset_disease/val',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),   
    layers.Dense(3, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(train_data.class_indices)
class_indices = train_data.class_indices

class_weight = {}
for class_name, index in class_indices.items():
    if class_name == 'TB':
        class_weight[index] = 4
    else:
        class_weight[index] = 1
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight
)

model.save("model/disease_model.h5")