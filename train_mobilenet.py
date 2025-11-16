from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
)

train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_gen = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Base Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train and store history
history1 = model.fit(train_gen, epochs=20, validation_data=test_gen)

# Unfreeze top layers for fine-tuning
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_gen, epochs=10, validation_data=test_gen)

model.save('waste_classifier_mobilenet.h5')
print("✅ MobileNetV2 Model trained and saved.")

# -------------------------------
# 📊 Plot Accuracy Graph
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 📉 Plot Loss Graph
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history1.history['loss'] + history2.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
