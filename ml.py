import os
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern
from skimage.filters import sobel


def extract_features(image_path):
    size = os.path.getsize(image_path) / 1024

    # Chargement OpenCV
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    # Composantes couleurs
    r_channel, g_channel, b_channel = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    r, g, b = r_channel.mean(), g_channel.mean(), b_channel.mean()
    r_std, g_std, b_std = r_channel.std(), g_channel.std(), b_channel.std()

    # Niveau de gris
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Contrast
    contrast = gray.std()

    # Luminance
    luminance_array = 0.2126 * r_channel + 0.7152 * g_channel + 0.0722 * b_channel
    luminance = luminance_array.mean()

    # Densité des bords
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height)

    # Saturation
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()

    # Entropie (informations)
    entropy = shannon_entropy(gray)

    # Texture par LBP
    lbp = local_binary_pattern(gray, P=8, R=1.0, method="uniform")
    texture_lbp = lbp.std()

    # Énergie des bords (via Sobel)
    sobel_edges = sobel(gray)
    edge_energy = np.sum(sobel_edges ** 2)

    return (
        width, height, float(size),
        int(r), int(g), int(b),
        float(contrast), float(saturation), float(luminance),
        float(edge_density), float(entropy), float(texture_lbp), float(edge_energy)
    )

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
"""
for filename in os.listdir("Data/train/no_label"):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join("Data/train/no_label", filename)
        features = extract_features(path)
        pred = clf.predict([features])[0]
        print(f"{filename} => {'Pleine' if pred == 1 else 'Vide'}")
"""
y_true = []
y_pred = []

for label_dir, label in [("clean", 0), ("dirty", 1)]:
    folder = os.path.join("Data/train/with_label", label_dir)
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            pred = clf.predict([features])[0]
            y_pred.append(pred)
            y_true.append(label)
            print(f"{filename} => {'Pleine' if pred == 1 else 'Vide'}")

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pleine', 'Vide'], yticklabels=['Pleine', 'Vide'])

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Par exemple, afficher le premier arbre de la forêt
estimator = clf.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(estimator,
          max_depth=3,         # Limite la profondeur affichée (exemple 3)
          feature_names=[
            "width", "height", "size",
            "r", "g", "b",
            "contrast", "saturation", "luminance",
            "edge_density", "entropy", "texture_lbp", "edge_energy"
          ],
          class_names=["Vide", "Pleine"],
          filled=True,         # Colore selon la classe majoritaire dans la feuille
          rounded=True,
          fontsize=10)
plt.show()
