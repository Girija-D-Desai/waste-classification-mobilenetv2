import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("waste_classifier_mobilenet.h5")

# Define class labels (update these if your dataset differs)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Create main window
root = tk.Tk()
root.title("Waste Classifier")
root.geometry("500x600")
root.configure(bg="#f0f0f0")

# Heading
Label(root, text="Waste Classification System", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333").pack(pady=10)

# Image display area
img_label = Label(root, bg="#f0f0f0")
img_label.pack(pady=10)

# Prediction label
result_label = Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0")
result_label.pack(pady=20)


def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        display_image(file_path)
        predict_image(file_path)


def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk


def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    result_label.config(text=f"Predicted Waste Type: {predicted_class}", fg="green")


# Upload button
Button(root, text="Upload Image", command=upload_image, font=("Arial", 12, "bold"),
       bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)

# Exit button
Button(root, text="Exit", command=root.destroy, font=("Arial", 12, "bold"),
       bg="#f44336", fg="white", padx=10, pady=5).pack(pady=10)

root.mainloop()
