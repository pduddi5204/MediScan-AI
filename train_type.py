from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.6,1.4],
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    'dataset_type/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = train_datagen.flow_from_directory(
    'dataset_type/val',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])


model.compile(
    optimizer=Adam(learning_rate=0.0001),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, validation_data=val_data, epochs=10)

model.save("model/type_model.h5")