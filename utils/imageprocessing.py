import cv2
import numpy as np
import os
import random
from flask import request
from utils.decisiontree import load_tree
from utils.classification import classify_image, calculate_seuils_from_db
from utils.featuresextraction import extract_features
from models.models import TrashImage, ClassificationRule, Settings, User, update_rule, db
from flask import current_app as app

def crop_img(image_path, marge_ratio=0.3, min_size=50, debug=False, output_dir="uploads/crop"):
    """Détecte et recadre automatiquement une poubelle dans une image"""
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    os.makedirs(output_dir, exist_ok=True)
    original_height, original_width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Détection améliorée du vert
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Nettoyage du masque
    kernel = np.ones((7, 7), np.uint8)
    mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    # Détection des contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_path
    # Sélection du meilleur contour
    best_contour = None
    best_score = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filtres de taille
        if w < 50 or h < 50:
            continue  
        area = cv2.contourArea(cnt)
        aspect_ratio = h / float(w)
        # Critères pour une poubelle
        if aspect_ratio < 0.8 or aspect_ratio > 2.5:
            continue
        # Score basé sur l'aire et la forme
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        # Bonus pour forme rectangulaire
        rect_area = w * h
        extent = area / rect_area
        score = area * solidity * extent
        if score > best_score:
            best_score = score
            best_contour = cnt
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        # Ajout de marge
        marge_x = int(w * marge_ratio)
        marge_y = int(h * marge_ratio)
        x1 = max(0, x - marge_x)
        y1 = max(0, y - marge_y)
        x2 = min(original_width, x + w + marge_x)
        y2 = min(original_height, y + h + marge_y)
        cropped = img_rgb[y1:y2, x1:x2]
        # Vérification de la taille minimale
        if cropped.shape[0] >= min_size and cropped.shape[1] >= min_size:
            crop_filename = os.path.basename(image_path).replace('.', '_crop.')
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            return crop_path
    return image_path

def add_img(img):
    if img and allowed_file(img.filename):
        filename = img.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        crop_filepath = crop_img(filepath)
        (width, height, size, r, g, b, contrast, saturation, 
         luminosity, edge_density, entropy, texture_variance, edge_energy,
         top_variance, top_entropy, hist_peaks, dark_ratio, bright_ratio, 
         color_uniformity, circle_count) = extract_features(crop_filepath)
        annotation = request.form.get('annotation')
        lat = random.uniform(48.5, 49)
        lng = random.uniform(2,2.6)
        if annotation not in ['Pleine', 'Vide', 'pleine', 'vide']:
            is_auto = annotation
            tree=load_tree()
            annotation = classify_image(size, r, g, b, width, height, 
                contrast, saturation, luminosity, 
                edge_density, entropy, texture_variance, edge_energy,
                top_variance, top_entropy, hist_peaks, dark_ratio, 
                bright_ratio, color_uniformity, circle_count, tree)
        else:
            is_auto = 'Manual'
        new_image = TrashImage(
            filename=filename,
            link=crop_filepath,
            annotation=annotation.capitalize(),
            width=width,
            height=height,
            filesize_kb=round(size, 2),
            avg_color_r=int(r),
            avg_color_g=int(g),
            avg_color_b=int(b),
            contrast=contrast,
            saturation=saturation,
            luminosity=luminosity,
            edge_density=edge_density,
            entropy=entropy,
            texture_variance=texture_variance,
            edge_energy=edge_energy,
            top_variance=top_variance,
            top_entropy=top_entropy,
            hist_peaks=hist_peaks,
            dark_ratio=dark_ratio,
            bright_ratio=bright_ratio,
            color_uniformity=color_uniformity,
            circle_count=circle_count,
            type=is_auto,
            lat=lat,
            lng=lng
        )
        db.session.add(new_image)
        db.session.commit()
        settings = Settings.query.first()
        if settings.use_auto_rules:
            seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
            if seuils_plein and seuils_vide:
                for rule in ClassificationRule.query.all():
                    if rule.name in seuils_plein and rule.name in seuils_vide:
                        update_rule(rule.name, (seuils_plein[rule.name][0] + seuils_vide[rule.name][0]) / 2)     
    else:
        return "Format de fichier non supporté", 400
    
def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS