from flask import Flask, render_template, request, redirect
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from flask_sqlalchemy import SQLAlchemy
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from flask import send_from_directory
from flask_socketio import SocketIO, emit
import random
import time, threading
from datetime import datetime

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

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
        "luminance": [0,900,0],
        "edge_density": [0,900,0]
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
        "luminance": [0,900,0],
        "edge_density": [0,900,0]
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
        "luminance": [0,900,0],
        "edge_density": [0,900,0]
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
    contrast = db.Column(db.Integer)
    saturation = db.Column(db.Float)
    luminance = db.Column(db.Float)
    edge_density = db.Column(db.Float)
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
        "luminance": 0.05,
        "edge_density": 0.05
    }
    for name, value in seuils.items():
        if ClassificationRule.query.filter_by(name=name).first() is None:
            db.session.add(ClassificationRule(name=name, value=value))
        if Settings.query.first() is None:
            db.session.add(Settings(use_auto_rules=True))
            db.session.commit()
    db.session.commit()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    size = os.path.getsize(image_path) / 1024
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    np_img = np.array(img)
    r, g, b = np_img[:, :, 0].mean(), np_img[:, :, 1].mean(), np_img[:, :, 2].mean()
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    luminance_array = 0.2126 * np_img[:, :, 0] + 0.7152 * np_img[:, :, 1] + 0.0722 * np_img[:, :, 2]
    luminance = luminance_array.mean()
    contrast = gray.std()
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    edge_density = np.sum(edges > 0) / (width * height)
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    return width, height, float(size), int(r), int(g), int(b), float(contrast), float(saturation), float(luminance), float(edge_density)

def in_range(value, interval):
    return interval[1] <= value <= interval[2]

def classify_image(avg_r, filesize_kb, avg_g, avg_b, width, height, contrast, saturation, luminance, edge_density):
    # Récupération des seuils avec ta fonction get_rule
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
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    score = 0
    if avg_r < r_thresh: score += 1
    if avg_g < g_thresh: score += 1
    if avg_b < b_thresh: score += 1
    if height > height_thresh: score += 1
    if edge_density > edge_density_thresh: score += 2
    if contrast > contrast_thresh: score += 1
    if saturation < saturation_thresh: score += 1
    if luminance < luminance_thresh: score += 1
    if filesize_kb > filesize_thresh: score += 3
    if avg_r < r_thresh and saturation < saturation_thresh: score += 1
    return "Pleine" if score >= 5 else "Vide"




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

def add_img(img):
    if img and allowed_file(img.filename):
        filename = img.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        link = filepath
        width, height, size, r, g, b, contrast, saturation, luminance, edge_density = extract_features(filepath)
        auto_annotation = classify_image(r, size, g, b, width, height, contrast, saturation, luminance, edge_density)
        annotation = request.form.get('annotation')
        lat = random.uniform(48.5, 49)
        lng = random.uniform(2,2.6)
        

        if annotation not in ['Pleine', 'Vide', 'pleine', 'vide']:
            is_auto = annotation
            annotation = auto_annotation
        else:
            is_auto = 'Manual'

        new_image = TrashImage(
            filename=filename,
            link=link,
            annotation=annotation.capitalize(),  # Majuscule
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
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
        return redirect('/')
    
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('index.html', images=images)

@app.route('/stats')
def stats():
    pleines = TrashImage.query.filter_by(annotation='Pleine').count()
    vides = TrashImage.query.filter_by(annotation='Vide').count()

    fig, ax = plt.subplots()
    ax.bar(['Pleine', 'Vide'], [pleines, vides], color=['red', 'green'])
    ax.set_title("Distribution des annotations")

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return f'<img src="data:image/png;base64,{plot_url}" />'

@app.route('/dashboard')
def dashboard():
    data = TrashImage.query.all()
    labels = [img.filename for img in data]
    values = [img.filesize_kb for img in data]
    annotations = [img.annotation for img in data]
    return render_template("dashboard.html", labels=labels, values=values, annotations=annotations)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/home', methods=['GET', 'POST'])
def home():
    settings = Settings.query.first()

    if request.method == 'POST':
        for name in request.form:
            if name != "use_auto_rules":
                value = float(request.form[name])
                update_rule(name, value)
        settings.use_auto_rules = 'use_auto_rules' in request.form
        db.session.commit()
        return redirect('/home')
    
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    images = TrashImage.query.all()
    return render_template(
        'home.html', rules=rules,
        seuils=seuils,
        seuils_plein=seuils_plein,
        seuils_vide=seuils_vide,
        images=images,
        use_auto_rules=settings.use_auto_rules
    )

@app.route('/predict', methods=['GET', 'POST'])
def user():
    if request.method == 'POST':
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
        return redirect('/predict')
    
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('user.html', images=images)

socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print("Client connecté")
    images = TrashImage.query.all()
    for img in images:
        send_marker_update(img)

def send_marker_update(img):
    socketio.emit('update_marker', {'id': img.id, 'lat': img.lat, 'lng': img.lng})

def evaluer_precision():
    images = TrashImage.query.filter(TrashImage.annotation.in_(['Pleine', 'Vide']), TrashImage.type == 'Manual').all()
    bonnes_preds = 0
    total = len(images)

    for img in images:
        prediction = classify_image(
            img.avg_color_r, img.filesize_kb, img.avg_color_g, img.avg_color_b,
            img.width, img.height, img.contrast, img.saturation,
            img.luminance, img.edge_density
        )
        if prediction == img.annotation:
            bonnes_preds += 1

    if total == 0:
        return "Aucune image annotée pour évaluer la précision."
    else:
        precision = bonnes_preds / total
        return f"Précision de l'IA codée en dur : {precision * 100:.2f}% ({bonnes_preds}/{total} prédictions correctes)"

@app.route('/evaluer_precision')
def route_precision():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Récupération des seuils depuis la base (exemple)
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()

    # Préparation d'une liste pour construire le DataFrame
    rows = []
    for var in seuils.keys():
        # seuils global (moyenne, min, max)
        m, mi, ma = seuils[var]
        # seuils plein (moyenne, min, max)
        mp, mip, map_ = seuils_plein.get(var, (None, None, None))
        # seuils vide (moyenne, min, max)
        mv, miv, mav = seuils_vide.get(var, (None, None, None))
        rows.append({
            "Variable": var,
            "Global_Mean": m,
            "Global_Min": mi,
            "Global_Max": ma,
            "Plein_Mean": mp,
            "Plein_Min": mip,
            "Plein_Max": map_,
            "Vide_Mean": mv,
            "Vide_Min": miv,
            "Vide_Max": mav,
        })

    # Création du DataFrame
    df = pd.DataFrame(rows)

    print(df)

    # --- Graphique ---

    fig, ax = plt.subplots(figsize=(12, 6))

    variables = df["Variable"].tolist()
    indices = range(len(variables))

    def plot_interval(ax, indices, means, mins, maxs, label, color):
        for i, (m, mi, ma) in enumerate(zip(means, mins, maxs)):
            if m is None:
                continue
            ax.plot([mi, ma], [i, i], color=color, lw=6, alpha=0.6)
            ax.plot(m, i, 'o', color=color, label=label if i == 0 else "")

    plot_interval(
        ax, indices,
        df["Global_Mean"], df["Global_Min"], df["Global_Max"],
        label="Global", color="gray"
    )
    plot_interval(
        ax, indices,
        df["Plein_Mean"], df["Plein_Min"], df["Plein_Max"],
        label="Plein", color="green"
    )
    plot_interval(
        ax, indices,
        df["Vide_Mean"], df["Vide_Min"], df["Vide_Max"],
        label="Vide", color="red"
    )

    ax.set_yticks(indices)
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel("Valeurs des seuils")
    ax.set_title("Comparaison des intervalles de seuils par variable")

    ax.legend()
    plt.tight_layout()
    plt.show()

    return evaluer_precision()

def tableau_confusion():
    from collections import Counter
    images = TrashImage.query.filter(TrashImage.annotation.in_(['Pleine', 'Vide']), TrashImage.type == 'Manual').all()
    confusion = Counter()

    for img in images:
        pred = classify_image(
            img.avg_color_r, img.filesize_kb, img.avg_color_g, img.avg_color_b,
            img.width, img.height, img.contrast, img.saturation,
            img.luminance, img.edge_density
        )
        confusion[(img.annotation, pred)] += 1

    return f"""Tableau de confusion :
    Vérité terrain \\ Prédiction
    Pleine -> Pleine : {confusion[("Pleine", "Pleine")]}
    Pleine -> Vide   : {confusion[("Pleine", "Vide")]}
    Vide   -> Pleine : {confusion[("Vide", "Pleine")]}
    Vide   -> Vide   : {confusion[("Vide", "Vide")]}"""

@app.route('/tableau_confusion')
def route_tableau_confusion():
    return tableau_confusion() 

if __name__ == '__main__':
    socketio.run(app, debug=True)
    app.run(debug=True)
