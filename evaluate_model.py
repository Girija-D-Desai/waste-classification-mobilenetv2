import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the trained model
model = load_model("waste_classifier_mobilenet.h5")

# Path to your test dataset
TEST_DIR = r"C:\Users\srush\OneDrive\Desktop\Waste_Classifier_Project\dataset\test"

TARGET_SIZE = (224,224)

# Prepare the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=TARGET_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ⭐ ADD THIS PART — Evaluate accuracy
loss, accuracy = model.evaluate(test_gen)
print(f"\n📌 Test Accuracy: {accuracy * 100:.2f}%\n")

# Predict
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
