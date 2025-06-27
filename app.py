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
        "contrast": [0,900,0]
    }
    seuils_plein = {
        "avg_color_r": [0,900,0],
        "avg_color_g": [0,900,0],
        "avg_color_b": [0,900,0],
        "filesize_kb": [0,900,0],
        "width": [0,900,0],
        "height": [0,900,0],
        "contrast": [0,900,0]
    }
    seuils_vide = {
        "avg_color_r": [0,900,0],
        "avg_color_g": [0,900,0],
        "avg_color_b": [0,900,0],
        "filesize_kb": [0,900,0],
        "width": [0,900,0],
        "height": [0,900,0],
        "contrast": [0,900,0]
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
    annotation = db.Column(db.String(10), nullable=True)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    filesize_kb = db.Column(db.Float)
    avg_color_r = db.Column(db.Integer)
    avg_color_g = db.Column(db.Integer)
    avg_color_b = db.Column(db.Integer)
    contrast = db.Column(db.Integer)

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
        "contrast": 0.05
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
    return width, height, size, int(r), int(g), int(b), contrast

def classify_image(avg_r, filesize_kb, avg_g, avg_b, width, height, contrast):
    r_thresh = get_rule("avg_color_r", 100)
    g_thresh = get_rule("avg_color_g", 100)
    b_thresh = get_rule("avg_color_b", 100)
    filesize_thresh = get_rule("filesize_kb", 500)
    width_thresh = get_rule("width", 100)
    height_thresh = get_rule("height", 100)
    contrast_thresh = get_rule("contrast", 0.05)
    if (avg_r < r_thresh and avg_g < g_thresh and avg_b < b_thresh and filesize_kb > filesize_thresh and width > width_thresh and height > height_thresh and contrast > contrast_thresh):
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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Pas de fichier image", 400
        img = request.files['image']
        if img.filename == '':
            return "Aucun fichier sélectionné", 400
        if img and allowed_file(img.filename):
            filename = img.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            width, height, size, r, g, b, contrast = extract_features(filepath)
            auto_annotation = classify_image(r, size, g, b, width, height, contrast)
            annotation = request.form.get('annotation')
            
            if annotation not in ['Pleine', 'Vide', 'pleine', 'vide']:
                annotation = auto_annotation

            new_image = TrashImage(
                filename=filename,
                annotation=annotation.capitalize(),  # Majuscule
                width=width,
                height=height,
                filesize_kb=round(size, 2),
                avg_color_r=r,
                avg_color_g=g,
                avg_color_b=b,
                contrast=contrast
            )
            db.session.add(new_image)
            db.session.commit()
            seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
            if seuils_plein and seuils_vide:
                for rule in ClassificationRule.query.all():
                    if rule.name in seuils_plein and rule.name in seuils_vide:
                            update_rule(rule.name, (seuils_plein[rule.name][2] + seuils_vide[rule.name][2]) / 2)
            return redirect('/')
        else:
            return "Format de fichier non supporté", 400

    images = TrashImage.query.all()
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
        flash("Règles mises à jour avec succès.", "success")
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

if __name__ == '__main__':
    app.run(debug=True)
