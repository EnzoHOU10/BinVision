import cv2
import numpy as np
from matplotlib import pyplot as plt
import os 

def detect_poubelle_crop(image_path, marge_ratio=0.1, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image introuvable.")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Masques de couleur
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    meilleure_poubelle = None
    meilleure_score = -1
    best_debug_rect = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 30:
            continue

        area = cv2.contourArea(cnt)
        aspect_ratio = h / float(w)
        if aspect_ratio < 0.6 or aspect_ratio > 3.5:
            continue

        # Score "forme"
        score = area

        # Éviter les branches/feuillage → trop de contours fins
        roi_edges = edges[y:y+h, x:x+w]
        edge_density = np.sum(roi_edges > 0) / (w * h)
        if edge_density > 0.20:
            continue  # trop de petits détails

        # Bonus si présence de rouge ou blanc en haut du bloc
        bande_y1 = max(0, y - int(0.2 * h))
        bande_y2 = y
        bande_red = mask_red[bande_y1:bande_y2, x:x+w]
        bande_white = mask_white[bande_y1:bande_y2, x:x+w]
        score += np.sum(bande_red) + np.sum(bande_white)

        if score > meilleure_score:
            meilleure_score = score
            marge_x = int(w * marge_ratio)
            marge_y = int(h * marge_ratio)
            x1 = max(0, x - marge_x)
            y1 = max(0, y - marge_y)
            x2 = min(img.shape[1], x + w + marge_x)
            y2 = min(img.shape[0], y + h + marge_y)
            meilleure_poubelle = img_rgb[y1:y2, x1:x2]
            best_debug_rect = (x1, y1, x2, y2)

    # Si aucun bon contour, on prend le plus gros vert (fallback)
    if meilleure_poubelle is None and len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        marge_x = int(w * marge_ratio)
        marge_y = int(h * marge_ratio)
        x1 = max(0, x - marge_x)
        y1 = max(0, y - marge_y)
        x2 = min(img.shape[1], x + w + marge_x)
        y2 = min(img.shape[0], y + h + marge_y)
        meilleure_poubelle = img_rgb[y1:y2, x1:x2]
        best_debug_rect = (x1, y1, x2, y2)
        print("⚠️ Fallback: aucune zone parfaite, on prend la plus grande.")

    if meilleure_poubelle is not None:
        if debug and best_debug_rect:
            x1, y1, x2, y2 = best_debug_rect
            img_debug = img_rgb.copy()
            cv2.rectangle(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 3)
            plt.imshow(img_debug)
            plt.title("Zone détectée (poubelle + marge)")
            plt.axis("off")
            plt.show()
            plt.imshow(meilleure_poubelle)
            plt.title("Crop centré sur la poubelle")
            plt.axis("off")
            plt.show()
        return meilleure_poubelle

    print("❌ Aucune zone verte trouvée.", image_path)
    return None


# === Test ===
image_path = "./Data/train/no_label"
os.makedirs("output", exist_ok=True)
for image in os.listdir(image_path):
    if image.endswith(".jpg"):
        crop = detect_poubelle_crop(os.path.join(image_path, image), marge_ratio=0.15, debug=True)
        if crop is not None:
            cv2.imwrite(f"output/crop_poubelle_{image}", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))





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
        'contrast', 'saturation', 'luminosity', 'edge', 'entropy', 'texture_lbp', 'edge_energy'
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

def gini_impurity(y):
    """ Calcule l'impureté de Gini """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)

def meilleur_seuil(X_col, y):
    """ Trouve le meilleur seuil pour une colonne X_col par rapport aux étiquettes y """
    meilleurs = {"seuil": None, "gini": float("inf")}
    valeurs = np.unique(X_col)
    
    for i in range(len(valeurs) - 1):
        seuil = (valeurs[i] + valeurs[i+1]) / 2  # entre deux valeurs

        gauche = y[X_col <= seuil]
        droite = y[X_col > seuil]

        gini_g = gini_impurity(gauche)
        gini_d = gini_impurity(droite)

        gini_total = (len(gauche) * gini_g + len(droite) * gini_d) / len(y)

        if gini_total < meilleurs["gini"]:
            meilleurs["seuil"] = seuil
            meilleurs["gini"] = gini_total
    
    return meilleurs["seuil"]

def meilleur_split(X, y):
    """ Retourne la meilleure feature et son seuil """
    meilleur = {"feature": None, "seuil": None, "gini": float("inf")}
    
    for col in X.columns:
        seuil = meilleur_seuil(X[col].values, y)
        if seuil is None:
            continue
        
        gauche = y[X[col] <= seuil]
        droite = y[X[col] > seuil]
        gini_total = (
            len(gauche) * gini_impurity(gauche) +
            len(droite) * gini_impurity(droite)
        ) / len(y)
        
        if gini_total < meilleur["gini"]:
            meilleur.update({"feature": col, "seuil": seuil, "gini": gini_total})
    
    return meilleur


def manual():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

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
        'edge': img.edge,
        'entropy': img.entropy,
        'texture_lbp': img.texture_lbp,
        'edge_energy': img.edge_energy,
        'annotation': img.annotation
    } for img in imgs])

    X = df.drop("annotation", axis=1)
    y = df["annotation"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X_train, y_train)

    print("Précision sur test :", clf.score(X_test, y_test))

    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=["Vide", "Pleine"], filled=True)
    plt.show()


def construire_arbre(profondeur=0, max_profondeur=4):
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
        'edge': img.edge,
        'entropy': img.entropy,
        'texture_lbp': img.texture_lbp,
        'edge_energy': img.edge_energy,
        'annotation': img.annotation
    } for img in imgs])

    X = df.drop("annotation", axis=1)
    y = df["annotation"]
    # Condition d'arrêt : pureté ou profondeur atteinte
    if len(np.unique(y)) == 1 or profondeur == max_profondeur:
        return {"label": y.iloc[0], "leaf": True}
    
    split = meilleur_split(X, y)
    if split["feature"] is None:
        # Pas de split possible
        return {"label": y.mode()[0], "leaf": True}
    
    feature = split["feature"]
    seuil = split["seuil"]

    gauche_idx = X[feature] <= seuil
    droite_idx = X[feature] > seuil

    arbre = {
        "feature": feature,
        "seuil": seuil,
        "gini": split["gini"],
        "left": construire_arbre(X[gauche_idx], y[gauche_idx], profondeur+1, max_profondeur),
        "right": construire_arbre(X[droite_idx], y[droite_idx], profondeur+1, max_profondeur),
        "leaf": False
    }
    return arbre

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
import pprint
import pandas as pd

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
    edge = db.Column(db.Float)
    entropy = db.Column(db.Float, nullable=True)
    texture_lbp = db.Column(db.Float, nullable=True)
    edge_energy = db.Column(db.Float, nullable=True)
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
        "avg_color_r": 132.5,
        "avg_color_g": 149,
        "avg_color_b": 149.5,
        "filesize_kb": 500,
        "width": 1525,
        "height": 411.5,
        "contrast": 0.05,
        "saturation": 0.263,
        "luminosity": 0.457,
        "edge": 28713.5,
        "entropy": 7.651,
        "texture_lbp": 2.476,
        "edge_energy": 3202.53
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

def crop_img(image_path, marge_ratio=0.3, debug=False, output_dir="uploads/crop"):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meilleure_poubelle = None
    meilleure_score = -1
    best_debug_rect = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 30:
            continue
        area = cv2.contourArea(cnt)
        aspect_ratio = h / float(w)
        if aspect_ratio < 0.6 or aspect_ratio > 3.5:
            continue
        score = area
        roi_edges = edges[y:y+h, x:x+w]
        edge_density = np.sum(roi_edges > 0) / (w * h)
        if edge_density > 0.20:
            continue
        bande_y1 = max(0, y - int(0.2 * h))
        bande_y2 = y
        bande_red = mask_red[bande_y1:bande_y2, x:x+w]
        bande_white = mask_white[bande_y1:bande_y2, x:x+w]
        score += np.sum(bande_red) + np.sum(bande_white)
        if score > meilleure_score:
            meilleure_score = score
            marge_x = int(w * marge_ratio)
            marge_y = int(h * marge_ratio)
            x1 = max(0, x - marge_x)
            y1 = max(0, y - marge_y)
            x2 = min(img.shape[1], x + w + marge_x)
            y2 = min(img.shape[0], y + h + marge_y)
            meilleure_poubelle = img_rgb[y1:y2, x1:x2]
            best_debug_rect = (x1, y1, x2, y2)

    if meilleure_poubelle is None and len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        marge_x = int(w * marge_ratio)
        marge_y = int(h * marge_ratio)
        x1 = max(0, x - marge_x)
        y1 = max(0, y - marge_y)
        x2 = min(img.shape[1], x + w + marge_x)
        y2 = min(img.shape[0], y + h + marge_y)
        meilleure_poubelle = img_rgb[y1:y2, x1:x2]
        best_debug_rect = (x1, y1, x2, y2)

    if meilleure_poubelle is not None:
        os.makedirs(output_dir, exist_ok=True)
        crop_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(crop_path, cv2.cvtColor(meilleure_poubelle, cv2.COLOR_RGB2BGR))
        if debug and best_debug_rect:
            x1, y1, x2, y2 = best_debug_rect
            img_debug = img_rgb.copy()
            cv2.rectangle(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 3)
            plt.imshow(img_debug)
            plt.title("Zone détectée (poubelle + marge)")
            plt.axis("off")
            plt.show()
            plt.imshow(meilleure_poubelle)
            plt.title("Crop centré sur la poubelle")
            plt.axis("off")
            plt.show()
        return crop_path
    return image_path

def extract_features(image_path):
    filesize_kb = os.path.getsize(image_path) / 1024
    img_rgb = Image.open(image_path).convert('RGB')
    width, height = img_rgb.size
    img_np = np.array(img_rgb)
    r, g, b = int(np.mean(img_np[:, :, 0])), int(np.mean(img_np[:, :, 1])), int(np.mean(img_np[:, :, 2]))
    #luminosity = 0.299 * r + 0.587 * g + 0.114 * b
    img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    luminosity = np.mean(img_ycrcb[:, :, 0]) / 255
    rgb_mean = [r, g, b]
    #saturation = (max(rgb_mean) - min(rgb_mean)) / (sum(rgb_mean) + 1e-6)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    saturation = np.mean(img_hsv[:, :, 1]) / 255
    hist_r = (np.histogram(img_np[:, :, 0], bins=256, range=(0, 256))[0]).tolist()
    hist_g = (np.histogram(img_np[:, :, 1], bins=256, range=(0, 256))[0]).tolist()
    hist_b = (np.histogram(img_np[:, :, 2], bins=256, range=(0, 256))[0]).tolist()
    #contrast = int(img_np.max() - img_np.min())
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    contrast = np.std(img_gray)
    edges = cv2.Canny(img_gray, 50, 150)
    edge = np.sum(edges > 0)
    hist_luminance = (np.histogram(img_gray, bins=256, range=(0, 256))[0]).tolist()
    entropy = shannon_entropy(img_gray)
    lbp = local_binary_pattern(img_gray, P=8, R=1.0, method="uniform")
    texture_lbp = lbp.std()
    sobel_edges = sobel(img_gray)
    edge_energy = np.sum(sobel_edges ** 2)
    return (
        int(width), int(height), float(filesize_kb),
        int(r), int(g), int(b), hist_r, hist_g, hist_b,
        int(contrast), float(saturation), float(luminosity), hist_luminance,
        int(edge), float(entropy), float(texture_lbp), float(edge_energy)
    )

def add_img(img):
    if img and allowed_file(img.filename):
        filename = img.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        link = filepath
        crop_filepath = crop_img(filepath, marge_ratio=0.15, debug=False)
        (width, height, size, r, g, b, hist_r, hist_g, hist_b, contrast, saturation, luminosity, hist_luminance, edge, entropy, texture_lbp, edge_energy) = extract_features(filepath)
        entropy=entropy
        texture_lbp=texture_lbp
        edge_energy=edge_energy
        annotation = request.form.get('annotation')
        lat = random.uniform(48.5, 49)
        lng = random.uniform(2,2.6)
        
        if annotation not in ['Pleine', 'Vide', 'pleine', 'vide']:
            is_auto = annotation
            annotation = classify_image(size, r, g, b, hist_r, hist_g, hist_b, width, height, contrast, saturation, luminosity, hist_luminance, edge, entropy, texture_lbp, edge_energy)
        else:
            is_auto = 'Manual'

        new_image = TrashImage(
            filename=filename,
            link=link,
            annotation=annotation.capitalize(),
            width=width,
            height=height,
            filesize_kb=round(size, 2),
            avg_color_r=r,
            avg_color_g=g,
            avg_color_b=b,
            contrast=contrast,
            saturation=saturation,
            luminosity=luminosity,
            edge=edge,
            entropy=entropy,
            texture_lbp=texture_lbp,
            edge_energy=edge_energy,
            type=is_auto,
            lat=lat,
            lng=lng
        )
        db.session.add(new_image)
        db.session.commit()
        
        settings = Settings.query.first()
        if settings.use_auto_rules:
            seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
            newrule=construire_arbre(profondeur=0, max_profondeur=4)
            for rule in ClassificationRule.query.all():
                if rule.name == newrule.name:
                    update_rule(rule.name, newrule.value)

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
        "edge": [0,900,0],
        "entropy": [0,900,0],
        "texture_lbp": [0,900,0],
        "edge_energy": [0,900,0]
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
        "edge": [0,900,0],
        "entropy": [0,900,0],
        "texture_lbp": [0,900,0],
        "edge_energy": [0,900,0]
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
        "edge": [0,900,0],
        "entropy": [0,900,0],
        "texture_lbp": [0,900,0],
        "edge_energy": [0,900,0]
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

def classify_image(filesize_kb, avg_r, avg_g, avg_b, hist_r, hist_g, hist_b, width, height, contrast, saturation, luminosity, hist_luminance, edge, entropy, texture_lbp, edge_energy):
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    r_seuil = get_rule("avg_color_r")
    g_seuil = get_rule("avg_color_g")
    b_seuil = get_rule("avg_color_b")
    filesize_seuil = get_rule("filesize_kb")
    width_seuil = get_rule("width")
    height_seuil = get_rule("height")
    contrast_seuil = get_rule("contrast")
    saturation_seuil = get_rule("saturation")
    luminosity_seuil = get_rule("luminosity")
    edge_density_seuil = get_rule("edge_density")
    entropy_seuil = get_rule("entropy")
    texture_seuil = get_rule("texture_lbp")
    energy_seuil = get_rule("edge_energy")
    
    avg_color = (avg_r + avg_g + avg_b) / 3

    if height <= height_seuil:
        if edge_energy <= energy_seuil:
            if avg_b <= b_seuil:
                return "Vide"
            else:
                if edge <= edge_density_seuil:
                    return "Vide"
                else:
                    return "Pleine"
        else:
            return "Pleine"
    else:
        if width <= width_seuil:
            if entropy <= entropy_seuil:
                if avg_b <= b_seuil/2:
                    return "Vide"
                else:
                    return "Vide"
            else:
                if height <= height_seuil*2:
                    return "Vide"
                else:
                    return "Pleine"
        else:
            if saturation <= saturation_seuil:
                if avg_r <= r_seuil:
                    return "Vide"
                else:
                    return "Pleine"
            else:
                if saturation <= saturation_seuil*2:
                    return "Pleine"
                else:
                    return "Vide"

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

    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    return render_template('user.html', rules=rules,
        seuils=seuils,
        seuils_plein=seuils_plein,
        seuils_vide=seuils_vide, 
        images=images, 
        use_auto_rules=settings.use_auto_rules, 
        user=user)

@app.route('/admin', methods=['GET', 'POST'])
def settings():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if request.method == 'POST':
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('settings.html', images=images, user=user)

@app.route('/dashboard')
def dashboard():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    images = TrashImage.query.all()

    total_images = len(images)
    pleines = sum(1 for img in images if img.annotation == 'Pleine')
    vides = sum(1 for img in images if img.annotation == 'Vide')
    sizes = [img.filesize_kb for img in images]
    annotations = [img.annotation for img in images]

    r_values = [img.avg_color_r for img in images if img.avg_color_r is not None]
    g_values = [img.avg_color_g for img in images if img.avg_color_g is not None]
    b_values = [img.avg_color_b for img in images if img.avg_color_b is not None]

    return render_template(
        "dashboard.html",
        total_images=total_images,
        pleines=pleines,
        vides=vides,
        sizes=sizes,
        annotations=annotations,
        r_values=r_values,
        g_values=g_values,
        b_values=b_values,
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

if __name__ == '__main__':
    """
    with app.app_context():
        #ml_evaluate_model()
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
            'edge': img.edge,
            'entropy': img.entropy,
            'texture_lbp': img.texture_lbp,
            'edge_energy': img.edge_energy,
            'annotation': img.annotation
        } for img in imgs])

        X = df.drop("annotation", axis=1)
        y = df["annotation"]
        arbre = construire_arbre(X, y, max_profondeur=4)
        pprint.pprint(arbre)
        manual()
        """
    socketio.run(app, debug=True)
    app.run(debug=True)


