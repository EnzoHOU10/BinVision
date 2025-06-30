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
        "luminance": [0,900,0]
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
        "luminance": [0,900,0]
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
        "luminance": [0,900,0]
    }
    rules = ClassificationRule.query.all()
    images = TrashImage.query.all()
    cpt=0
    cpt_plein = 0
    cpt_vide = 0
    for image in images:
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
    auto_annotated = db.Column(db.Boolean, default=False)


class ClassificationRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Float, nullable=False)

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
        "luminance": 0.05
    }
    for name, value in seuils.items():
        if ClassificationRule.query.filter_by(name=name).first() is None:
            db.session.add(ClassificationRule(name=name, value=value))
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
    edges = cv2.Canny(gray, 100, 200)
    contrast = edges.sum() / (width * height)
    saturation = np.sqrt((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return width, height, float(size), int(r), int(g), int(b), float(contrast), float(saturation), float(luminance)

def classify_image(avg_r, filesize_kb, avg_g, avg_b, width, height, contrast, saturation, luminance):
    r_thresh = get_rule("avg_color_r")
    g_thresh = get_rule("avg_color_g")
    b_thresh = get_rule("avg_color_b")
    filesize_thresh = get_rule("filesize_kb")
    width_thresh = get_rule("width")
    height_thresh = get_rule("height")
    contrast_thresh = get_rule("contrast")
    saturation_thresh = get_rule("saturation")
    luminance_thresh = get_rule("luminance")
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    avg_rgb = (avg_r + avg_g + avg_b) / 3

    if contrast > contrast_thresh * 1.5:
        return "Pleine"
    if contrast < contrast_thresh * 0.5 and filesize_kb < filesize_thresh * 0.7:
        return "Vide"
    # Contraste très bas et couleurs homogènes → souvent vide
    if contrast < contrast_thresh * 0.7 and abs(avg_r - avg_g) < 10 and abs(avg_r - avg_b) < 10:
        return "Vide"
    if filesize_kb > filesize_thresh * 1.8 and avg_r < r_thresh and avg_g < g_thresh and avg_b < b_thresh:
        return "Pleine"
    # Image très claire, contraste faible, petite taille → vide
    if avg_rgb > 230 and contrast < contrast_thresh and filesize_kb < filesize_thresh:
        return "Vide"
    if avg_r > r_thresh * 1.2 and avg_g > g_thresh * 1.2 and avg_b > b_thresh * 1.2:
        return "Vide"
    # Image très grande + lourde + contrastée → pleine
    if width > width_thresh * 1.3 and height > height_thresh * 1.3 and filesize_kb > filesize_thresh * 1.5 and contrast > contrast_thresh:
        return "Pleine"
    
    def distance_to_profile(profile):
        return (
            abs(avg_r - profile["avg_color_r"][0]) +
            abs(avg_g - profile["avg_color_g"][0]) +
            abs(avg_b - profile["avg_color_b"][0]) +
            abs(filesize_kb - profile["filesize_kb"][0]) +
            abs(width - profile["width"][0]) +
            abs(height - profile["height"][0]) +
            abs(contrast - profile["contrast"][0]) +
            abs(saturation - profile["saturation"][0]) +
            abs(luminance - profile["luminance"][0])
        )

    if seuils_plein and seuils_vide:
        dist_plein = distance_to_profile(seuils_plein)
        dist_vide = distance_to_profile(seuils_vide)
        return "Pleine" if dist_plein < dist_vide else "Vide"
    
    return "Pleine"



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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        images = request.files.getlist('image')
        print("Images reçues:", images)
        for img in images:
            print("Traitement de l'image:", img)
            if img and allowed_file(img.filename):
                filename = img.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(filepath)
                link = filepath
                width, height, size, r, g, b, contrast, saturation, luminance = extract_features(filepath)
                auto_annotation = classify_image(r, size, g, b, width, height, contrast, saturation, luminance)
                annotation = request.form.get('annotation')
                
                if annotation not in ['Pleine', 'Vide', 'pleine', 'vide']:
                    annotation = auto_annotation
                    is_auto = True
                else:
                    is_auto = False

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
                    auto_annotated=is_auto
                )
                db.session.add(new_image)
                db.session.commit()
                if not is_auto:
                    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
                    if seuils_plein and seuils_vide:
                        for rule in ClassificationRule.query.all():
                            if rule.name in seuils_plein and rule.name in seuils_vide:
                                update_rule(rule.name, (seuils_plein[rule.name][0] + seuils_vide[rule.name][0]) / 2)     
            else:
                return "Format de fichier non supporté", 400
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

@app.route('/rules', methods=['GET', 'POST'])
def rules():
    if request.method == 'POST':
        for name in request.form:
            value = float(request.form[name])
            rule = ClassificationRule.query.filter_by(name=name).first()
            if rule:
                rule.value = value
            else:
                rule = ClassificationRule(name=name, value=value)
                db.session.add(rule)
        db.session.commit()
        return redirect('/rules')
    
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    print("Seuils:", seuils)
    print("Seuils Plein:", seuils_plein)
    print("Seuils Vide:", seuils_vide)
    return render_template('rules.html', rules=rules, seuils=seuils, seuils_plein=seuils_plein, seuils_vide=seuils_vide)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        for name in request.form:
            value = float(request.form[name])
            rule = ClassificationRule.query.filter_by(name=name).first()
            if rule:
                rule.value = value
            else:
                rule = ClassificationRule(name=name, value=value)
                db.session.add(rule)
        db.session.commit()
        return redirect('/rules')
    
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    print("Seuils:", seuils)
    print("Seuils Plein:", seuils_plein)
    print("Seuils Vide:", seuils_vide)
    return render_template('home.html', rules=rules, seuils=seuils, seuils_plein=seuils_plein, seuils_vide=seuils_vide)

socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print("Client connecté")
    send_marker_update()

def send_marker_update():
    socketio.emit('update_marker', {'id': 'tour_eiffel', 'lat': 48.8584, 'lng': 2.2945})

if __name__ == '__main__':
    socketio.run(app, debug=True)
    app.run(debug=True)
