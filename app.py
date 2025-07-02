import os
import io
import base64
import random
from datetime import datetime

from flask import Flask, render_template, request, redirect, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

from PIL import Image
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')  # backend non‑interactif pour serveur
import matplotlib.pyplot as plt

# ---------------------------
# Machine‑Learning imports
# ---------------------------
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# ---------------------------
# Configuration
# ---------------------------
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "model.pkl"

# ---------------------------
# Flask / SQLAlchemy / SocketIO
# ---------------------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# crée le dossier upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ======================================================
# Base de données
# ======================================================
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
    luminance = db.Column(db.Float)
    edge_density = db.Column(db.Float)
    type = db.Column(db.String(10), nullable=True)  # "Manual" ou "Auto"
    lat = db.Column(db.Float, nullable=True)
    lng = db.Column(db.Float, nullable=True)

class ClassificationRule(db.Model):  # Ancienne heuristique (encore utilisée en secours)
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Float, nullable=False)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    use_auto_rules = db.Column(db.Boolean, default=True)  # si True, on ré‑entraîne automatiquement

# ======================================================
# Utilitaires divers
# ======================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(image_path):
    """Retourne toutes les features numériques exploitées par le modèle"""
    size = os.path.getsize(image_path) / 1024  # kB

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    np_img = np.array(img)

    # Moyennes RGB
    r, g, b = np_img[:, :, 0].mean(), np_img[:, :, 1].mean(), np_img[:, :, 2].mean()

    # Luminance, contraste, densité de contours
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    luminance_array = 0.2126 * np_img[:, :, 0] + 0.7152 * np_img[:, :, 1] + 0.0722 * np_img[:, :, 2]
    luminance = luminance_array.mean()
    contrast = gray.std()

    # Canny adaptatif
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = np.sum(edges > 0) / (width * height)

    # Saturation moyenne (HSV)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()

    return (
        width,
        height,
        float(size),
        int(r),
        int(g),
        int(b),
        float(contrast),
        float(saturation),
        float(luminance),
        float(edge_density),
    )

# ======================================================
# Heuristique d'origine (fallback)
# ======================================================

DEFAULT_RULES = {
    "avg_color_r": 100,
    "avg_color_g": 100,
    "avg_color_b": 100,
    "filesize_kb": 500,
    "width": 100,
    "height": 100,
    "contrast": 0.05,
    "saturation": 0.05,
    "luminance": 0.05,
    "edge_density": 0.05,
}


def get_rule(name, default=0):
    rule = ClassificationRule.query.filter_by(name=name).first()
    return rule.value if rule else default


def heuristic_classify_image(
    avg_r,
    filesize_kb,
    avg_g,
    avg_b,
    width,
    height,
    contrast,
    saturation,
    luminance,
    edge_density,
):
    # Récupération des seuils stockés
    r_thresh = get_rule("avg_color_r")
    g_thresh = get_rule("avg_color_g")
    b_thresh = get_rule("avg_color_b")
    filesize_thresh = get_rule("filesize_kb")
    width_thresh = get_rule("width")
    height_thresh = get_rule("height")
    contrast_thresh = get_rule("contrast")
    saturation_thresh = get_rule("saturation")
    luminance_thresh = get_rule("luminance")
    edge_density_thresh = get_rule("edge_density")

    score = 0
    if avg_r < r_thresh:
        score += 1
    if avg_g < g_thresh:
        score += 1
    if avg_b < b_thresh:
        score += 1
    if height > height_thresh:
        score += 1
    if edge_density > edge_density_thresh:
        score += 2
    if contrast > contrast_thresh:
        score += 1
    if saturation < saturation_thresh:
        score += 1
    if luminance < luminance_thresh:
        score += 1
    if filesize_kb > filesize_thresh:
        score += 3
    if avg_r < r_thresh and saturation < saturation_thresh:
        score += 1

    return "Pleine" if score >= 5 else "Vide"

# ======================================================
# Machine‑learning tabulaire
# ======================================================

def fetch_dataset():
    """Récupère toutes les images annotées manuellement (Pleine / Vide) et renvoie un DataFrame"""
    data = TrashImage.query.filter(
        TrashImage.type == "Manual", TrashImage.annotation.in_(["Pleine", "Vide"])
    ).all()
    rows = [
        {
            "avg_color_r": img.avg_color_r,
            "avg_color_g": img.avg_color_g,
            "avg_color_b": img.avg_color_b,
            "filesize_kb": img.filesize_kb,
            "width": img.width,
            "height": img.height,
            "contrast": img.contrast,
            "saturation": img.saturation,
            "luminance": img.luminance,
            "edge_density": img.edge_density,
            "label": 1 if img.annotation == "Pleine" else 0,
        }
        for img in data
    ]
    return pd.DataFrame(rows)


def train_and_save():
    """(Ré)‑entraîne le modèle, sauvegarde sur disque et renvoie la précision de validation."""
    df = fetch_dataset()
    if df.empty or df["label"].nunique() < 2:
        return 0.0  # pas assez de données

    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return acc


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# Modèle chargé en mémoire (global)
model = None

# ======================================================
# Classification unique exposée au reste de l'appli
# ======================================================

def classify_image(
    avg_r,
    filesize_kb,
    avg_g,
    avg_b,
    width,
    height,
    contrast,
    saturation,
    luminance,
    edge_density,
):
    """Décide Pleine/Vide en privilégiant le modèle ML s'il existe."""
    global model
    if model is not None:
        import numpy as np

        X = np.array(
            [
                [
                    avg_r,
                    avg_g,
                    avg_b,
                    filesize_kb,
                    width,
                    height,
                    contrast,
                    saturation,
                    luminance,
                    edge_density,
                ]
            ]
        )
        proba = model.predict_proba(X)[0, 1]
        return "Pleine" if proba >= 0.5 else "Vide"

    # fallback heuristique
    return heuristic_classify_image(
        avg_r,
        filesize_kb,
        avg_g,
        avg_b,
        width,
        height,
        contrast,
        saturation,
        luminance,
        edge_density,
    )

# ======================================================
# Initialisation de la base + règles par défaut + modèle
# ======================================================
with app.app_context():
    db.create_all()

    # Règles par défaut si la table est vide
    for name, value in DEFAULT_RULES.items():
        if ClassificationRule.query.filter_by(name=name).first() is None:
            db.session.add(ClassificationRule(name=name, value=value))

    if Settings.query.first() is None:
        db.session.add(Settings(use_auto_rules=True))

    db.session.commit()

    # Chargement modèle ML
    model = load_model()

# ======================================================
# Helpers applicatifs
# ======================================================

def add_img(img):
    if img and allowed_file(img.filename):
        filename = img.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img.save(filepath)

        (
            width,
            height,
            size,
            r,
            g,
            b,
            contrast,
            saturation,
            luminance,
            edge_density,
        ) = extract_features(filepath)

        auto_annotation = classify_image(
            r,
            size,
            g,
            b,
            width,
            height,
            contrast,
            saturation,
            luminance,
            edge_density,
        )

        annotation = request.form.get("annotation")
        if annotation not in ["Pleine", "Vide", "pleine", "vide"]:
            img_type = "Auto"
            annotation = auto_annotation
        else:
            img_type = "Manual"

        # coordonnées aléatoires (à remplacer par vraies valeurs si dispo)
        lat = random.uniform(48.5, 49)
        lng = random.uniform(2, 2.6)

        new_image = TrashImage(
            filename=filename,
            link=filepath,
            annotation=annotation.capitalize(),
            width=width,
            height=height,
            filesize_kb=round(size, 2),
            avg_color_r=r,
            avg_color_g=g,
            avg_color_b=b,
            contrast=contrast,
            saturation=saturation,
            luminance=luminance,
            edge_density=edge_density,
            type=img_type,
            lat=lat,
            lng=lng,
        )
        db.session.add(new_image)
        db.session.commit()

        # Ré‑entraîne automatiquement si l'option est activée et qu'il s'agit d'une nouvelle annotation manuelle
        settings = Settings.query.first()
        global model
        if settings and settings.use_auto_rules and img_type == "Manual":
            acc = train_and_save()
            model = load_model()
            print(f"Modèle ré‑entraîné automatiquement (val_acc={acc*100:.2f}%)")
    else:
        return "Format de fichier non supporté", 400

# ======================================================
# Routes Flask
# ======================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        images = request.files.getlist("image")
        for img in images:
            add_img(img)
        return redirect("/")

    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template("index.html", images=images)


@app.route("/stats")
def stats():
    pleines = TrashImage.query.filter_by(annotation="Pleine").count()
    vides = TrashImage.query.filter_by(annotation="Vide").count()

    fig, ax = plt.subplots()
    ax.bar(["Pleine", "Vide"], [pleines, vides], color=["red", "green"])
    ax.set_title("Distribution des annotations")

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_buf.seek(0)
    plot_url = base64.b64encode(img_buf.getvalue()).decode()

    return f'<img src="data:image/png;base64,{plot_url}" />'


@app.route("/dashboard")
def dashboard():
    data = TrashImage.query.all()
    labels = [img.filename for img in data]
    values = [img.filesize_kb for img in data]
    annotations = [img.annotation for img in data]
    return render_template("dashboard.html", labels=labels, values=values, annotations=annotations)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/home", methods=["GET", "POST"])
def home():
    settings = Settings.query.first()

    if request.method == "POST":
        for name in request.form:
            if name != "use_auto_rules":
                value = float(request.form[name])
                rule = ClassificationRule.query.filter_by(name=name).first()
                if rule:
                    rule.value = value
        settings.use_auto_rules = "use_auto_rules" in request.form
        db.session.commit()
        return redirect("/home")

    rules = ClassificationRule.query.all()
    images = TrashImage.query.all()
    return render_template(
        "home.html",
        rules=rules,
        images=images,
        use_auto_rules=settings.use_auto_rules if settings else False,
    )


@app.route("/predict", methods=["GET", "POST"])
def user():
    if request.method == "POST":
        images = request.files.getlist("image")
        for img in images:
            add_img(img)
        return redirect("/predict")

    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template("user.html", images=images)


# -------------- ML utils exposés -------------

@app.route("/train_model")
def train_model_route():
    acc = train_and_save()
    global model
    model = load_model()
    return f"Modèle ré‑entraîné. Précision de validation : {acc*100:.2f}%"


# -------------- Évaluation -------------------

def evaluer_precision():
    images = TrashImage.query.filter(
        TrashImage.annotation.in_(["Pleine", "Vide"]), TrashImage.type == "Manual"
    ).all()
    bonnes_preds = 0
    total = len(images)

    for img in images:
        prediction = classify_image(
            img.avg_color_r,
            img.filesize_kb,
            img.avg_color_g,
            img.avg_color_b,
            img.width,
            img.height,
            img.contrast,
            img.saturation,
            img.luminance,
            img.edge_density,
        )
        if prediction == img.annotation:
            bonnes_preds += 1

    if total == 0:
        return "Aucune image annotée pour évaluer la précision."
    else:
        precision = bonnes_preds / total
        return f"Précision de l'IA : {precision * 100:.2f}% ({bonnes_preds}/{total} prédictions correctes)"


@app.route("/evaluer_precision")
def route_precision():
    return evaluer_precision()


# -------------- Tableau de confusion ---------

from collections import Counter


def tableau_confusion():
    images = TrashImage.query.filter(
        TrashImage.annotation.in_(["Pleine", "Vide"]), TrashImage.type == "Manual"
    ).all()
    confusion = Counter()

    for img in images:
        pred = classify_image(
            img.avg_color_r,
            img.filesize_kb,
            img.avg_color_g,
            img.avg_color_b,
            img.width,
            img.height,
            img.contrast,
            img.saturation,
            img.luminance,
            img.edge_density,
        )
        confusion[(img.annotation, pred)] += 1

    return (
        """Tableau de confusion :\n"
        "Vérité terrain \\ Prédiction\n"
        f"Pleine -> Pleine : {confusion[("Pleine", "Pleine")]}\n"
        f"Pleine -> Vide   : {confusion[("Pleine", "Vide")]}\n"
        f"Vide   -> Pleine : {confusion[("Vide", "Pleine")]}\n"
        f"Vide   -> Vide   : {confusion[("Vide", "Vide")]}"""
    )


@app.route("/tableau_confusion")
def route_tableau_confusion():
    return tableau_confusion()

# ======================================================
# SocketIO temps‑réel
# ======================================================

@socketio.on("connect")
def handle_connect():
    print("Client connecté")
    images = TrashImage.query.all()
    for img in images:
        socketio.emit("update_marker", {"id": img.id, "lat": img.lat, "lng": img.lng})

# ======================================================
# Lancement
# ======================================================

if __name__ == "__main__":
    socketio.run(app, debug=True)
