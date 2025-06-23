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

class ClassificationRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()
    default_rules = {
        "avg_r_threshold": 100,
        "avg_g_threshold": 100,
        "avg_b_threshold": 100,
        "filesize_threshold_kb": 500,
        "min_width": 100,
        "min_height": 100,
        "contrast_threshold": 0.05
    }

    for name, value in default_rules.items():
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
    return width, height, size, int(r), int(g), int(b)

def classify_image(avg_r, filesize_kb, avg_g, avg_b, width, height, contrast):
    r_thresh = get_rule("avg_r_threshold", 100)
    g_thresh = get_rule("avg_g_threshold", 100)
    b_thresh = get_rule("avg_b_threshold", 100)
    filesize_thresh = get_rule("filesize_threshold_kb", 500)
    width_thresh = get_rule("min_width", 100)
    height_thresh = get_rule("min_height", 100)
    contrast_thresh = get_rule("contrast_threshold", 0.05)
    if (avg_r < r_thresh and avg_g < g_thresh and avg_b < b_thresh and filesize_kb > filesize_thresh and width > width_thresh and height > height_thresh and contrast > contrast_thresh):
        return "Pleine"
    else:
        return "Vide"

def get_rule(name, default=0):
    rule = ClassificationRule.query.filter_by(name=name).first()
    return rule.value if rule else default


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Pas de fichier image", 400
        img = request.files['image']
        if img.filename == '':
            return "Aucun fichier sélectionné", 400
        if img and allowed_file(img.filename):
            # Nom de fichier sécurisé et unique
            filename = img.filename
            # Tu peux rajouter un UUID ou timestamp si tu veux éviter les collisions
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)

            # Extraire les caractéristiques après sauvegarde
            width, height, size, r, g, b = extract_features(filepath)
            gray = cv2.cvtColor(np.array(Image.open(filepath).convert('RGB')), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contrast = edges.sum() / (width * height)

            auto_annotation = classify_image(r, size, g, b, width, height, contrast)

            # Récupérer annotation manuelle (peut être vide)
            annotation = request.form.get('annotation')
            # Si annotation manuelle non fournie, utiliser automatique
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
                avg_color_b=b
            )
            db.session.add(new_image)
            db.session.commit()
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
    return render_template('rules.html', rules=rules)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
