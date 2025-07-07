import os
import numpy as np
import cv2
from PIL import Image
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

def extract_features(image_path):
    """Extrait toutes les caractéristiques d'une image"""
    # Taille du fichier
    filesize_kb = os.path.getsize(image_path) / 1024  
    # Ouverture de l'image
    img_rgb = Image.open(image_path).convert('RGB')
    width, height = img_rgb.size
    img_np = np.array(img_rgb)
    # Couleurs moyennes
    r, g, b = np.mean(img_np[:, :, 0]), np.mean(img_np[:, :, 1]), np.mean(img_np[:, :, 2])
    # Luminosité, saturation, contraste
    luminosity = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = (np.max(img_np, axis=2) - np.min(img_np, axis=2)).mean()
    contrast = np.std(img_np)
    # Conversion en niveaux de gris
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Détection des bords
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height)
    # Entropie
    entropy = shannon_entropy(img_gray)
    # Texture (LBP)
    lbp = local_binary_pattern(img_gray, P=8, R=1.0, method="uniform")
    texture_variance = np.var(lbp)
    # Énergie des bords (Sobel)
    sobel_edges = sobel(img_gray)
    edge_energy = np.sum(sobel_edges ** 2)
    # Analyse de la partie supérieure
    top = img_gray[-height//3:, :]
    top_variance = np.var(top)
    top_entropy = shannon_entropy(top)
    # Histogramme et pics
    hist_luminance = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_peaks = len([i for i in range(1, 255) if hist_luminance[i] > hist_luminance[i-1] and hist_luminance[i] > hist_luminance[i+1]])
    # Ratios sombre/clair
    dark_ratio = np.sum(img_gray < 100) / (width * height)
    bright_ratio = np.sum(img_gray > 180) / (width * height)
    # Uniformité des couleurs
    color_uniformity = 1 / (1 + np.std([r, g, b]))
    # Détection de cercles
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)
    circle_count = len(circles[0]) if circles is not None else 0
    return (
        int(width), int(height), float(filesize_kb),
        int(r), int(g), int(b),
        float(contrast), float(saturation), float(luminosity),
        float(edge_density), float(entropy), float(texture_variance), float(edge_energy),
        float(top_variance), float(top_entropy), int(hist_peaks),
        float(dark_ratio), float(bright_ratio), float(color_uniformity), int(circle_count)
    )