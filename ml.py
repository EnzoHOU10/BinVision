import os
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(image_path):
    size = os.path.getsize(image_path) / 1024
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    np_img = np.array(img)
    r, g, b = np_img[:, :, 0].mean(), np_img[:, :, 1].mean(), np_img[:, :, 2].mean()
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contrast = edges.sum() / (width * height)
    return [width, height, size, int(r), int(g), int(b), contrast]

X = []
y = []

for label_dir, label in [("clean", 0), ("dirty", 1)]:
    folder = os.path.join("Data/train/with_label", label_dir)
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            X.append(features)
            y.append(label)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

print("Modèle entraîné et sauvegardé.")

for filename in os.listdir("Data/train/no_label"):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join("Data/train/no_label", filename)
        features = extract_features(path)
        pred = clf.predict([features])[0]
        print(f"{filename} => {'Pleine' if pred == 1 else 'Vide'}")

"""
accuracy = accuracy_score()
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pleine', 'Vide'], yticklabels=['Pleine', 'Vide'])

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
"""