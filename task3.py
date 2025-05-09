import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === Path to dataset ===
base_path = r'C:/Users/BHAKTI/OneDrive/Desktop/ML Internship/Recogniser/animals'

# === Configuration ===
image_size = 64  # Resize to 64x64
max_images_per_class = 1000  # Adjust based on availability

# === Load images ===
def load_images(folder, label, max_images):
    data, labels = [], []
    count = 0
    for filename in os.listdir(folder):
        if count >= max_images:
            break
        try:
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            img = cv2.resize(img, (image_size, image_size))
            data.append(img.flatten())  # Flatten image
            labels.append(label)
            count += 1
        except Exception as e:
            print(f"Skipped {filename}: {e}")
            continue
    return data, labels

print("Loading images...")
cat_folder = os.path.join(base_path, 'cat')
dog_folder = os.path.join(base_path, 'dog')

# Fix: Ensure dog_data and cat_data are loaded correctly
cat_data, cat_labels = load_images(cat_folder, label=0, max_images=max_images_per_class)
dog_data, dog_labels = load_images(dog_folder, label=1, max_images=max_images_per_class)

X = np.array(cat_data + dog_data)
y = np.array(cat_labels + dog_labels)

# === Train/Test Split ===
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Feature Scaling (Important for SVM) ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Train SVM ===
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# === Predict and Evaluate ===
print("Evaluating...")
y_pred = svm.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# === Optional: Visualize Predictions ===
def show_predictions(X_orig, y_true, y_pred, num=10):
    plt.figure(figsize=(15, 4))
    for i in range(num):
        img = X_orig[i].reshape(image_size, image_size, 3)
        plt.subplot(2, 5, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Pred: {'Dog' if y_pred[i] else 'Cat'}\nTrue: {'Dog' if y_true[i] else 'Cat'}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Convert back from scaled data to image-like uint8 format ===
X_vis = (X_test * scaler.scale_ + scaler.mean_).astype(np.uint8)

# === Show predictions ===
show_predictions(X_vis, y_test, y_pred)
