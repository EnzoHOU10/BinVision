from flask import Flask, render_template, request, redirect, session
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from flask_sqlalchemy import SQLAlchemy
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import send_from_directory
from flask_socketio import SocketIO
import random
from datetime import datetime
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from werkzeug.security import generate_password_hash, check_password_hash
import seaborn as sns
import pandas as pd
from sqlalchemy.orm import class_mapper
import json

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "projetbinvision2025"

db = SQLAlchemy(app)
socketio = SocketIO(app)

class TrashImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    link = db.Column(db.String(200), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    annotation = db.Column(db.String(10), nullable=True)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    filesize_kb = db.Column(db.Float)
    avg_color_r = db.Column(db.Integer)
    avg_color_g = db.Column(db.Integer)
    avg_color_b = db.Column(db.Integer)
    contrast = db.Column(db.Float)
    saturation = db.Column(db.Float)
    luminosity = db.Column(db.Float)
    edge_density = db.Column(db.Float)
    entropy = db.Column(db.Float, nullable=True)
    texture_variance = db.Column(db.Float, nullable=True)
    edge_energy = db.Column(db.Float, nullable=True)
    top_variance = db.Column(db.Float, nullable=True)
    top_entropy = db.Column(db.Float, nullable=True)
    hist_peaks = db.Column(db.Integer, nullable=True)
    dark_ratio = db.Column(db.Float, nullable=True)
    bright_ratio = db.Column(db.Float, nullable=True)
    color_uniformity = db.Column(db.Float, nullable=True)
    circle_count = db.Column(db.Integer, nullable=True)
    type = db.Column(db.String(10), nullable=True)
    lat = db.Column(db.Float, nullable=True)
    lng = db.Column(db.Float, nullable=True)
    
class ClassificationRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Float, nullable=False)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    use_auto_rules = db.Column(db.Boolean, default=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')

def create_admin_account():
    admin_username = "admin"
    admin_email = "admin@binvision.com"
    admin_password = "12345"
    admin_role = "admin"
    if not User.query.filter_by(email=admin_email).first():
        hashed_pw = generate_password_hash(admin_password)
        admin = User(username=admin_username, email=admin_email, password=hashed_pw, role=admin_role)
        db.session.add(admin)
        db.session.commit()

with app.app_context():
    db.create_all()
    seuils = {
        "avg_color_r": 100,
        "avg_color_g": 100,
        "avg_color_b": 100,
        "filesize_kb": 500,
        "width": 100,
        "height": 100,
        "contrast": 0.05,
        "saturation": 0.05,
        "luminosity": 0.05,
        "edge_density": 0.05,
        "entropy": 5.0,
        "texture_variance": 0.5,
        "edge_energy": 1000.0,
        "top_variance": 0.5,
        "top_entropy": 4.0,
        "hist_peaks": 10,
        "dark_ratio": 0.3,
        "bright_ratio": 0.2,
        "color_uniformity": 0.5,
        "circle_count": 3,
    }
    for name, value in seuils.items():
        if ClassificationRule.query.filter_by(name=name).first() is None:
            db.session.add(ClassificationRule(name=name, value=value))
        if Settings.query.first() is None:
            db.session.add(Settings(use_auto_rules=True))
            db.session.commit()
    db.session.commit()
    create_admin_account()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def crop_img(image_path, marge_ratio=0.3, min_size=100, debug=False, output_dir="uploads/crop"):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
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
        
        # Sélection du best contour
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
                output_dir = os.path.dirname(image_path)
                crop_filename = os.path.basename(image_path).replace('.', '_crop.')
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                return crop_path
        
        return image_path
        
    except Exception as e:
        print(f"Erreur lors du crop: {e}")
        return image_path

def extract_features(image_path):
    try:
        filesize_kb = os.path.getsize(image_path) / 1024
        img_rgb = Image.open(image_path).convert('RGB')
        width, height = img_rgb.size
        img_np = np.array(img_rgb)
        r, g, b = np.mean(img_np[:, :, 0]), np.mean(img_np[:, :, 1]), np.mean(img_np[:, :, 2])
        luminosity = 0.299 * r + 0.587 * g + 0.114 * b
        saturation = (np.max(img_np, axis=2) - np.min(img_np, axis=2)).mean()
        contrast = np.std(img_np)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        entropy = shannon_entropy(img_gray)
        lbp = local_binary_pattern(img_gray, P=8, R=1.0, method="uniform")
        texture_variance = np.var(lbp)
        sobel_edges = sobel(img_gray)
        edge_energy = np.sum(sobel_edges ** 2)
        top = img_gray[-height//3:, :]
        top_variance = np.var(top)
        top_entropy = shannon_entropy(top)
        hist_luminance = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist_peaks = len([i for i in range(1, 255) if hist_luminance[i] > hist_luminance[i-1] and hist_luminance[i] > hist_luminance[i+1]])
        dark_ratio = np.sum(img_gray < 100) / (width * height)
        bright_ratio = np.sum(img_gray > 180) / (width * height)
        color_uniformity = 1 / (1 + np.std([r, g, b]))
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        circle_count = len(circles[0]) if circles is not None else 0
        return (
            int(width), int(height), float(filesize_kb),
            int(r), int(g), int(b),
            float(contrast), float(saturation), float(luminosity),
            float(edge_density), float(entropy), float(texture_variance), float(edge_energy),
            float(top_variance), float(top_entropy), int(hist_peaks),
            float(dark_ratio), float(bright_ratio), float(color_uniformity), int(circle_count)
        )
        
    except Exception as e:
        print(f"Erreur extraction features: {e}")
        return None

def add_img(img):
    if img and allowed_file(img.filename):
        filename = img.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        link = filepath
        crop_filepath = crop_img(filepath)
        (width, height, size, r, g, b, contrast, saturation, 
         luminosity, edge_density, entropy, texture_variance, edge_energy,
         top_variance, top_entropy, hist_peaks, dark_ratio, bright_ratio, 
         color_uniformity, circle_count) = extract_features(filepath)
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
    
def calculate_seuils_from_db():
    seuils = {
        "avg_color_r": [0,900,0],
        "avg_color_g": [0,900,0],
        "avg_color_b": [0,900,0],
        "filesize_kb": [0,900,0],
        "width": [0,900,0],
        "height": [0,900,0],
        "contrast": [0,900,0],
        "saturation": [0,900,0],
        "luminosity": [0,900,0],
        "edge_density": [0,900,0],
        "entropy": [0,900,0],
        "texture_variance": [0,900,0],
        "edge_energy": [0,900,0],
        "top_variance": [0,900,0],
        "top_entropy": [0,900,0],
        "hist_peaks": [0,900,0],
        "dark_ratio": [0,900,0],
        "bright_ratio": [0,900,0],
        "color_uniformity": [0,900,0],
        "circle_count": [0,900,0],
    }
    seuils_plein = {
        "avg_color_r": [0,900,0],
        "avg_color_g": [0,900,0],
        "avg_color_b": [0,900,0],
        "filesize_kb": [0,900,0],
        "width": [0,900,0],
        "height": [0,900,0],
        "contrast": [0,900,0],
        "saturation": [0,900,0],
        "luminosity": [0,900,0],
        "edge_density": [0,900,0],
        "entropy": [0,900,0],
        "texture_variance": [0,900,0],
        "edge_energy": [0,900,0],
        "top_variance": [0,900,0],
        "top_entropy": [0,900,0],
        "hist_peaks": [0,900,0],
        "dark_ratio": [0,900,0],
        "bright_ratio": [0,900,0],
        "color_uniformity": [0,900,0],
        "circle_count": [0,900,0],
    }
    seuils_vide = {
        "avg_color_r": [0,900,0],
        "avg_color_g": [0,900,0],
        "avg_color_b": [0,900,0],
        "filesize_kb": [0,900,0],
        "width": [0,900,0],
        "height": [0,900,0],
        "contrast": [0,900,0],
        "saturation": [0,900,0],
        "luminosity": [0,900,0],
        "edge_density": [0,900,0],
        "entropy": [0,900,0],
        "texture_variance": [0,900,0],
        "edge_energy": [0,900,0],
        "top_variance": [0,900,0],
        "top_entropy": [0,900,0],
        "hist_peaks": [0,900,0],
        "dark_ratio": [0,900,0],
        "bright_ratio": [0,900,0],
        "color_uniformity": [0,900,0],
        "circle_count": [0,900,0],
    }
    rules = ClassificationRule.query.all()
    images = TrashImage.query.all()
    cpt=0
    cpt_plein = 0
    cpt_vide = 0
    for image in images:
        if image.type == 'Manual':
            cpt+=1
            if image.annotation == 'Pleine':
                cpt_plein += 1
            else:
                cpt_vide += 1
            for rule in rules:
                seuils[rule.name][0] += getattr(image, rule.name, None)
                if getattr(image, rule.name, None) < seuils[rule.name][1]:
                    seuils[rule.name][1] = getattr(image, rule.name, None)
                if getattr(image, rule.name, None) > seuils[rule.name][2]:
                    seuils[rule.name][2] = getattr(image, rule.name, None)
                if image.annotation == 'Pleine':
                    seuils_plein[rule.name][0] += getattr(image, rule.name, None)
                    if getattr(image, rule.name, None) < seuils_plein[rule.name][1]:
                        seuils_plein[rule.name][1] = getattr(image, rule.name, None)
                    if getattr(image, rule.name, None) > seuils_plein[rule.name][2]:
                        seuils_plein[rule.name][2] = getattr(image, rule.name, None)
                else:
                    seuils_vide[rule.name][0] += getattr(image, rule.name, None)
                    if getattr(image, rule.name, None) < seuils_vide[rule.name][1]:
                        seuils_vide[rule.name][1] = getattr(image, rule.name, None)
                    if getattr(image, rule.name, None) > seuils_vide[rule.name][2]:
                        seuils_vide[rule.name][2] = getattr(image, rule.name, None)
    if cpt > 0:
        for rule in rules:
                seuils[rule.name][0] = seuils[rule.name][0] / cpt
        if cpt_plein > 0 and cpt_vide > 0:
            for rule in rules:
                    seuils_plein[rule.name][0] = seuils_plein[rule.name][0] / cpt_plein
            for rule in rules:
                    seuils_vide[rule.name][0] = seuils_vide[rule.name][0] / cpt_vide
            return seuils, seuils_plein, seuils_vide
        elif cpt_plein > 0 and cpt_vide == 0:
            for rule in rules:
                seuils_plein[rule.name][0] = seuils_plein[rule.name][0] / cpt_plein
            return seuils, seuils_plein, None
        else:
            for rule in rules:
                seuils_vide[rule.name][0] = seuils_vide[rule.name][0] / cpt_vide
            return seuils, None, seuils_vide
    else:
        return None, None, None

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)

def best_seuil(X_col, y):
    bests = {"seuil": None, "gini": float("inf")}
    valeurs = np.unique(X_col)
    for i in range(len(valeurs) - 1):
        seuil = (valeurs[i] + valeurs[i+1]) / 2
        gauche = y[X_col <= seuil]
        droite = y[X_col > seuil]
        gini_g = gini_impurity(gauche)
        gini_d = gini_impurity(droite)
        gini_total = (len(gauche) * gini_g + len(droite) * gini_d) / len(y)
        if gini_total < bests["gini"]:
            bests["seuil"] = seuil
            bests["gini"] = gini_total
    return bests["seuil"]

def best_split(X, y):
    best = {"feature": None, "seuil": None, "gini": float("inf")}
    for col in X.columns:
        seuil = best_seuil(X[col].values, y)
        if seuil is None:
            continue
        gauche = y[X[col] <= seuil]
        droite = y[X[col] > seuil]
        gini_total = (
            len(gauche) * gini_impurity(gauche) +
            len(droite) * gini_impurity(droite)
        ) / len(y)
        
        if gini_total < best["gini"]:
            best.update({"feature": col, "seuil": seuil, "gini": gini_total})
    return best

def create_tree(X,y,profondeur=0, max_profondeur=5):
    if len(np.unique(y)) == 1 or profondeur == max_profondeur:
        return {"label": y.iloc[0], "leaf": True}
    
    split = best_split(X, y)
    if split["feature"] is None:
        return {"label": y.mode()[0], "leaf": True}
    
    feature = split["feature"]
    seuil = split["seuil"]

    gauche_idx = X[feature] <= seuil
    droite_idx = X[feature] > seuil

    tree = {
        "feature": feature,
        "seuil": seuil,
        "gini": split["gini"],
        "left": create_tree(X[gauche_idx], y[gauche_idx], profondeur+1, max_profondeur),
        "right": create_tree(X[droite_idx], y[droite_idx], profondeur+1, max_profondeur),
        "leaf": False
    }
    return tree

def save_tree(tree, filename="cache/tree_cache/decision_tree.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(tree, f)

def load_tree(filename="cache/tree_cache/decision_tree.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def classify_image(filesize_kb, avg_r, avg_g, avg_b, width, height, contrast, saturation, luminosity, edge_density, entropy, texture_variance, edge_energy,top_variance, top_entropy, hist_peaks, dark_ratio, bright_ratio, color_uniformity, circle_count, tree):
    if tree is None:
        tree = load_tree()
        if tree is None:
            return "Vide"
    features = {
        "width": width,
        "height": height,
        "filesize_kb": filesize_kb,
        "avg_color_r": avg_r,
        "avg_color_g": avg_g,
        "avg_color_b": avg_b,
        "contrast": contrast,
        "saturation": saturation,
        "luminosity": luminosity,
        "edge_density": edge_density,
        "entropy": entropy,
        "texture_variance": texture_variance,
        "edge_energy": edge_energy,
        "top_variance": top_variance,
        "top_entropy": top_entropy,
        "hist_peaks": hist_peaks,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "color_uniformity": color_uniformity,
        "circle_count": circle_count,
    }
    def predict(features, tree):
        if tree.get("leaf", False):
            return tree["label"]
        if features[tree["feature"]] <= tree["seuil"]:
            return predict(features, tree["left"])
        else:
            return predict(features, tree["right"])
    
    return predict(features,tree)

def get_rule(name, default=0):
    rule = ClassificationRule.query.filter_by(name=name).first()
    return rule.value if rule else default

def update_rule(name, value):
    rule = ClassificationRule.query.filter_by(name=name).first()
    if rule:
        rule.value = value
    else:
        rule = ClassificationRule(name=name, value=value)
        db.session.add(rule)
    db.session.commit()

def generate_matplotlib(output_dir='static/matplotlib'):
    images = TrashImage.query.all()
    os.makedirs(output_dir, exist_ok=True)
    data = []
    for img in images:
        row = {col.key: getattr(img, col.key) for col in class_mapper(TrashImage).columns}
        data.append(row)
    df = pd.DataFrame(data)
    subsets = {
        "totalgraph": df,
        "pleinegraph": df[df['annotation'] == 'Pleine'],
        "videgraph": df[df['annotation'] == 'Vide'],
    }
    filenames = {
        "totalgraph": "totalgraph.png",
        "pleinegraph": "pleinegraph.png",
        "videgraph": "videgraph.png",
    }
    paths = []
    for key, subset_df in subsets.items():
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Analyse des images - {key.capitalize()}", fontsize=16)

        sns.histplot(subset_df['filesize_kb'].dropna(), bins=30, color="#245311", ax=axs[0,0])
        axs[0,0].set_title("Distribution Taille fichiers (KB)")
        axs[0,0].set_xlabel("Taille (KB)")
        axs[0,0].set_ylabel("Nombre d'images")

        sns.histplot(subset_df['contrast'].dropna(), bins=30, color='#B22222', ax=axs[0,1])
        axs[0,1].set_title("Distribution Contraste")
        axs[0,1].set_xlabel("Contraste")
        axs[0,1].set_ylabel("Nombre d'images")

        sc = axs[1,0].scatter(
            subset_df['luminosity'], 
            subset_df['saturation'], 
            c=subset_df['filesize_kb'], cmap='viridis', alpha=0.7
        )
        axs[1,0].set_title("Luminosité vs Saturation (couleur = taille fichier)")
        axs[1,0].set_xlabel("Luminosité")
        axs[1,0].set_ylabel("Saturation")
        cbar = fig.colorbar(sc, ax=axs[1,0])
        cbar.set_label('Taille fichier (KB)')

        metrics = ['texture_variance', 'entropy']
        means = subset_df[metrics].mean()
        stds = subset_df[metrics].std()

        axs[1,1].bar(metrics, means, yerr=stds, capsize=8, color=['black', '#B22222'])
        axs[1,1].set_title("Moyenne ± écart-type: Texture LBP & Entropie")
        axs[1,1].set_ylabel("Valeurs")
        axs[1,1].set_ylim(0, max(means + stds)*1.2 if not means.empty else 1)
        axs[1,1].set_facecolor('#f7f7f7')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        full_path = os.path.join(output_dir, filenames[key])
        web_path = f"../{output_dir}/{filenames[key]}".replace("//", "/")
        fig.savefig(full_path, bbox_inches='tight')
        plt.close(fig)
        paths.append(web_path)
    return paths

def ml_evaluate_model():
    """
    Entraîne un modèle ML sur les images annotées manuellement (type='Manual')
    et teste sur les images annotées automatiquement (type='Auto').
    Affiche accuracy, recall, F1, matrice de confusion.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    features = [
        'width', 'height', 'filesize_kb', 'avg_color_r', 'avg_color_g', 'avg_color_b',
        'contrast', 'saturation', 'luminosity', 'edge_density', 'entropy', 'texture_variance', 'edge_energy'
    ]

    def get_X_y(imgs):
        X, y = [], []
        for img in imgs:
            row = [getattr(img, f) for f in features]
            if None in row or img.annotation not in ['Pleine', 'Vide']:
                continue
            X.append(row)
            y.append(1 if img.annotation == 'Pleine' else 0)
        return X, y

    # Récupère les données
    train_imgs = TrashImage.query.filter_by(type='Manual').all()
    test_imgs = TrashImage.query.filter_by(type='Auto').all()

    # Convertit en X/y
    X_train, y_train = get_X_y(train_imgs)
    X_test, y_test = get_X_y(test_imgs)

    if not X_train or not X_test:
        print("Pas assez de données valides pour entraîner ou tester.")
        return

    # Entraîne le modèle
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Affiche les résultats
    print("\n--- Évaluation du modèle ML ---")
    print(classification_report(y_test, y_pred, target_names=['Vide', 'Pleine']))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
    print("Accuracy :", accuracy_score(y_test, y_pred))

    # Affichage graphique de la matrice
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=['Vide', 'Pleine'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - Test sur Auto")
    plt.show()

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/account', methods=['GET', 'POST'])
def account():
    message = ""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'edit' and user:
            new_username = request.form.get('edit_username', '').strip()
            new_email = request.form.get('edit_email', '').strip()
            new_password = request.form.get('edit_password', '').strip()
            if new_email and new_email != user.email and User.query.filter_by(email=new_email).first():
                message = "Cet email est déjà utilisé."
            else:
                if new_username:
                    user.username = new_username
                if new_email:
                    user.email = new_email
                if new_password:
                    user.password = generate_password_hash(new_password)
                db.session.commit()
                message = "Profil mis à jour."
        else:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()
            if action == "register":
                if User.query.filter_by(email=email).first():
                    message = "Email déjà pris."
                else:
                    if not username:
                        temp_user = User(username="temp", email=email, password=generate_password_hash(password))
                        db.session.add(temp_user)
                        db.session.flush()
                        temp_user.username = f"user{temp_user.id}"
                        db.session.commit()
                        message = "Inscription réussie. Connectez-vous."
                        user = None
                        return render_template('account.html', message=message, user=None)
                    else:
                        hashed_pw = generate_password_hash(password)
                        user = User(username=username, email=email, password=hashed_pw)
                        db.session.add(user)
                        db.session.commit()
                        message = "Inscription réussie. Connectez-vous."
            elif action == "login":
                user = User.query.filter_by(email=email).first()
                if user and check_password_hash(user.password, password):
                    session['user_id'] = user.id
                    return redirect('/account')
                else:
                    message = "Identifiants incorrects."
            user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return render_template('account.html', message=message, user=user)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/account')

@app.route('/', methods=['GET', 'POST'])
def home():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    images = TrashImage.query.filter_by(annotation='Pleine').all()
    images_conv=[]
    for img in images:
        images_conv.append({
            'id': img.id,
            'lat': img.lat,
            'lng': img.lng,
        })
    return render_template(
        'home.html', rules=rules,
        seuils=seuils,
        seuils_plein=seuils_plein,
        seuils_vide=seuils_vide,
        images=images_conv,
        user=user
    )

@app.route('/predict', methods=['GET', 'POST'])
def user():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])

    settings = Settings.query.first()
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        if form_type == 'rules':
            for name in request.form:
                if name not in ["use_auto_rules", "form_type"]:
                    try:
                        value = float(request.form[name])
                        update_rule(name, value)
                    except ValueError:
                        continue
            settings.use_auto_rules = 'use_auto_rules' in request.form
            db.session.commit()
            return redirect('/predict')
        elif form_type == 'upload':
            images = request.files.getlist('image')
            for img in images:
                add_img(img)
            return redirect('/predict')
        
    images = TrashImage.query.filter_by(annotation='Auto').all()
    images_conv=[]
    for img in images:
        images_conv.append({
            'id': img.id,
            'filename': img.filename,
            'link': img.link,
            'date': img.date,
            'annotation': img.annotation,
            'width': img.width,
            'height': img.height,
            'filesize_kb': img.filesize_kb,
            'avg_color_r': img.avg_color_r,
            'avg_color_g': img.avg_color_g,
            'avg_color_b': img.avg_color_b,
            'contrast': img.contrast,
            'saturation': img.saturation,
            'luminosity': img.luminosity,
            'edge_density': img.edge_density,
            'entropy': img.entropy,
            'texture_variance': img.texture_variance,
            'edge_energy': img.edge_energy,
            'top_variance': img.top_variance,
            'top_entropy': img.top_entropy,
            'hist_peaks': img.hist_peaks,
            'dark_ratio': img.dark_ratio,
            'bright_ratio': img.bright_ratio,
            'color_uniformity': img.color_uniformity,
            'circle_count': img.circle_count,
            'type': img.type,
            'lat': img.lat,
            'lng': img.lng
        })
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    return render_template('user.html', rules=rules,
        seuils=seuils,
        seuils_plein=seuils_plein,
        seuils_vide=seuils_vide, 
        images=images_conv, 
        use_auto_rules=settings.use_auto_rules, 
        user=user)

@app.route('/admin', methods=['GET', 'POST'])
def settings():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if user==None:
        return redirect('/account')
    if request.method == 'POST':
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
            imgs = TrashImage.query.filter_by(type='Manual').all()
            df = pd.DataFrame([{  
                'width': img.width,
                'height': img.height,
                'filesize_kb': img.filesize_kb,
                'avg_color_r': img.avg_color_r,
                'avg_color_g': img.avg_color_g,
                'avg_color_b': img.avg_color_b,
                'contrast': img.contrast,
                'saturation': img.saturation,
                'luminosity': img.luminosity,
                'edge_density': img.edge_density,
                'entropy': img.entropy,
                'texture_variance': img.texture_variance,
                'edge_energy': img.edge_energy,     
                'top_variance': img.top_variance,
                'top_entropy': img.top_entropy,
                'hist_peaks': img.hist_peaks,
                'dark_ratio': img.dark_ratio,
                'bright_ratio': img.bright_ratio,
                'color_uniformity': img.color_uniformity,
                'circle_count': img.circle_count,
                'annotation': img.annotation
            } for img in imgs])
            X = df.drop("annotation", axis=1)
            y = df["annotation"]
            save_tree(create_tree(X,y))
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('settings.html', images=images, user=user)

@app.route('/dashboard')
def dashboard():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])

    rules = ClassificationRule.query.all()
    images = TrashImage.query.all()
    images_conv=[]
    for img in images:
        images_conv.append({
            'id': img.id,
            'filename': img.filename,
            'link': img.link,
            'date': img.date,
            'annotation': img.annotation,
            'width': img.width,
            'height': img.height,
            'filesize_kb': img.filesize_kb,
            'avg_color_r': img.avg_color_r,
            'avg_color_g': img.avg_color_g,
            'avg_color_b': img.avg_color_b,
            'contrast': img.contrast,
            'saturation': img.saturation,
            'luminosity': img.luminosity,
            'edge_density': img.edge_density,
            'entropy': img.entropy,
            'texture_variance': img.texture_variance,
            'edge_energy': img.edge_energy,     
            'top_variance': img.top_variance,
            'top_entropy': img.top_entropy,
            'hist_peaks': img.hist_peaks,
            'dark_ratio': img.dark_ratio,
            'bright_ratio': img.bright_ratio,
            'color_uniformity': img.color_uniformity,
            'circle_count': img.circle_count,       
            'type': img.type,
            'lat': img.lat,
            'lng': img.lng
        })
    files = ['totalgraph.png', 'pleinegraph.png', 'videgraph.png']
    files_exist = all(os.path.exists(os.path.join("static/matplotlib", f)) for f in files)
    if files_exist:
        paths = [f"{"../static/matplotlib"}/{f}" for f in files]
    else:
        paths = generate_matplotlib()
    return render_template(
        'dashboard.html', 
        rules=rules,
        images=images_conv,
        paths=paths,
        user=user
    )

@app.route('/delete_marker/<int:marker_id>', methods=['DELETE'])
def delete_marker(marker_id):
    marker = TrashImage.query.get(marker_id)
    if marker:
        db.session.delete(marker)
        db.session.commit()
        return {'success': True, 'message': 'Marker supprimé'}, 200
    else:
        return {'success': False, 'message': 'Marker introuvable'}, 404

@socketio.on('connect')
def handle_connect():
    print("Client connecté")

from math import ceil
def test_manual():
        imgs = TrashImage.query.filter_by(type='Manual').all()
        df = pd.DataFrame([{
            'width': img.width,
            'height': img.height,
            'filesize_kb': img.filesize_kb,
            'avg_color_r': img.avg_color_r,
            'avg_color_g': img.avg_color_g,
            'avg_color_b': img.avg_color_b,
            'contrast': img.contrast,
            'saturation': img.saturation,
            'luminosity': img.luminosity,
            'edge_density': img.edge_density,
            'entropy': img.entropy,
            'texture_variance': img.texture_variance,
            'edge_energy': img.edge_energy,
            'top_variance': img.top_variance,
            'top_entropy': img.top_entropy,
            'hist_peaks': img.hist_peaks,
            'dark_ratio': img.dark_ratio,
            'bright_ratio': img.bright_ratio,
            'color_uniformity': img.color_uniformity,
            'circle_count': img.circle_count,
            'annotation': img.annotation
        } for img in imgs])

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = ceil(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        X_train = train_df.drop("annotation", axis=1)
        y_train = train_df["annotation"]
        tree = create_tree(X_train, y_train)
        X_test = test_df.drop("annotation", axis=1)
        y_test = test_df["annotation"]
        y_pred = []
        for _, row in X_test.iterrows():
            pred = classify_image(
                row["filesize_kb"], row["avg_color_r"], row["avg_color_g"], row["avg_color_b"],
                row["width"], row["height"], row["contrast"], row["saturation"], row["luminosity"],
                row["edge_density"], row["entropy"], row["texture_variance"], row["edge_energy"],
                row["top_variance"], row["top_entropy"], row["hist_peaks"],
                row["dark_ratio"], row["bright_ratio"], row["color_uniformity"], row["circle_count"],
                tree
            )
            y_pred.append(pred)
        correct = sum(p == t for p, t in zip(y_pred, y_test))
        total = len(y_test)
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy (arbre de décision fait maison) : {accuracy:.3f}")

if __name__ == '__main__':
    with app.app_context():
        print(test_manual())
    socketio.run(app, debug=True)
    app.run(debug=True)
