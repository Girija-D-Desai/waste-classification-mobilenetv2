import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import Tk, filedialog

# Load model
model = tf.keras.models.load_model("waste_classifier_mobilenet.h5")

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Open file chooser
root = Tk()
root.withdraw()  # hide main window
img_path = filedialog.askopenfilename(title="Select an image file")

if img_path:
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    print(f"\n🧠 Predicted Waste Type: {predicted_class}")
else:
    print("No file selected.")
