# diagnose_counts.py
import os

train_dir = r"C:\Users\srush\OneDrive\Desktop\Waste_Classifier_Project\dataset\train"
test_dir  = r"C:\Users\srush\OneDrive\Desktop\Waste_Classifier_Project\dataset\test"

def counts(path):
    print(f"\nDirectory: {path}")
    for c in sorted(os.listdir(path)):
        cpath = os.path.join(path, c)
        if os.path.isdir(cpath):
            imgs = [f for f in os.listdir(cpath) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            print(f"  {c}: {len(imgs)}")

print("TRAIN:")
counts(train_dir)
print("\nTEST:")
counts(test_dir)
