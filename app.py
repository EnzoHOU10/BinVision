from flask import Flask, render_template, request, redirect, session
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
from datetime import datetime
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from werkzeug.security import generate_password_hash, check_password_hash

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "projetbinvision2025"

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
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')

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
        "edge": 0.05,
        "entropy": 5.0,
        "texture_lbp": 0.5,
        "edge_energy": 1000.0

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
    filesize_kb = os.path.getsize(image_path) / 1024

    # Chargement OpenCV
    img_rgb = Image.open(image_path).convert('RGB')
    width, height = img_rgb.size

    # Composantes couleurs
    img_np = np.array(img_rgb)
    r, g, b = int(np.mean(img_np[:, :, 0])), int(np.mean(img_np[:, :, 1])), int(np.mean(img_np[:, :, 2]))

    #luminosité
    luminosity = 0.299 * r + 0.587 * g + 0.114 * b

    #saturation
    rgb_mean = [r, g, b]
    saturation = (max(rgb_mean) - min(rgb_mean)) / (sum(rgb_mean) + 1e-6)

    # Histogramme des couleurs
    hist_r = (np.histogram(img_np[:, :, 0], bins=256, range=(0, 256))[0]).tolist()
    hist_g = (np.histogram(img_np[:, :, 1], bins=256, range=(0, 256))[0]).tolist()
    hist_b = (np.histogram(img_np[:, :, 2], bins=256, range=(0, 256))[0]).tolist()
    
    # Contrast
    contrast = int(img_np.max() - img_np.min())

    # Détection de contours
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    edge = np.sum(edges > 0)
    
    # Histogramme de luminance
    hist_luminance = (np.histogram(img_gray, bins=256, range=(0, 256))[0]).tolist()

    # Entropie (informations)
    entropy = shannon_entropy(img_gray)

    # Texture par LBP
    lbp = local_binary_pattern(img_gray, P=8, R=1.0, method="uniform")
    texture_lbp = lbp.std()

    # Énergie des bords (via Sobel)
    sobel_edges = sobel(img_gray)
    edge_energy = np.sum(sobel_edges ** 2)

    return (
        width, height, float(filesize_kb),
        int(r), int(g), int(b),
        float(contrast), float(saturation), float(luminosity),
        float(edge), float(entropy), float(texture_lbp), float(edge_energy)
    )

def classify_image(avg_r, filesize_kb, avg_g, avg_b, width, height, contrast, saturation, luminosity, edge, entropy, texture_lbp, edge_energy):
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
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
    entropy_thresh = get_rule("entropy")
    texture_thresh = get_rule("texture_lbp")
    energy_thresh = get_rule("edge_energy")
    score = 0
    if height > height_thresh:
        score += 2
    if width > width_thresh:
        score += 1
    if filesize_kb > filesize_thresh:
        score += 2
    if edge > edge_density_thresh:
        score += 2
    if contrast > contrast_thresh:
        score += 1
    if entropy > entropy_thresh:
        score += 1
    if edge_energy > energy_thresh:
        score += 1
    if saturation < saturation_thresh:
        score += 1
    if luminosity < luminance_thresh:
        score += 1
    if avg_r < r_thresh:
        score += 1
    if avg_g < g_thresh:
        score += 1
    if avg_b < b_thresh:
        score += 1
    if texture_lbp < texture_thresh:
        score += 1

    def in_range(val, minv, maxv):
        return minv <= val <= maxv
    
    for name, val in [('height', height), ('width', width), ('filesize_kb', filesize_kb),('contrast', contrast), ('saturation', saturation), ('luminosity', luminosity), ('edge', edge), ('entropy', entropy), ('texture_lbp', texture_lbp), ('edge_energy', edge_energy), ('avg_color_r', avg_r), ('avg_color_g', avg_g), ('avg_color_b', avg_b)]:
        plein_min, plein_max = seuils_plein[name][1], seuils_plein[name][2]
        vide_min, vide_max = seuils_vide[name][1], seuils_vide[name][2]
        if in_range(val, plein_min, plein_max) and not in_range(val, vide_min, vide_max):
            score += 1  # fort indice "Pleine"
        if not in_range(val, plein_min, plein_max) and in_range(val, vide_min, vide_max):
            score -= 1

    # Décision finale : seuil à ajuster selon tes tests (ex : >=7)
    return "Pleine" if score >= 8 else "Vide"




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
        # Remplace l'appel par :
        (width, height, size, r, g, b, contrast, saturation, luminosity, edge, entropy, texture_lbp, edge_energy) = extract_features(filepath)
        # Puis ajoute ces lignes dans new_image :
        entropy=entropy
        texture_lbp=texture_lbp
        edge_energy=edge_energy

        annotation = request.form.get('annotation')
        lat = random.uniform(48.5, 49)
        lng = random.uniform(2,2.6)
        
        
        if annotation not in ['Pleine', 'Vide', 'pleine', 'vide']:
            is_auto = annotation
            annotation = classify_image(r, size, g, b, width, height, contrast, saturation, luminosity, edge, entropy, texture_lbp, edge_energy)
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
            if seuils_plein and seuils_vide:
                for rule in ClassificationRule.query.all():
                    if rule.name in seuils_plein and rule.name in seuils_vide:
                        update_rule(rule.name, (seuils_plein[rule.name][0] + seuils_vide[rule.name][0]) / 2)     
    else:
        return "Format de fichier non supporté", 400
    
@app.route('/', methods=['GET', 'POST'])
def index():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if request.method == 'POST':
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('index.html', images=images, user=user)

@app.route('/dashboard')
def dashboard():
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
        b_values=b_values
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/home', methods=['GET', 'POST'])
def home():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])

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
        use_auto_rules=settings.use_auto_rules,
        user=user
    )

@app.route('/predict', methods=['GET', 'POST'])
def user():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if request.method == 'POST':
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
        return redirect('/predict')
    
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('user.html', images=images, user=user)

from flask import flash, session

@app.route('/account', methods=['GET', 'POST'])
def account():
    message = ""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form['action']
        if action == "register":
            if User.query.filter_by(username=username).first():
                message = "Nom d'utilisateur déjà pris."
            else:
                hashed_pw = generate_password_hash(password)
                user = User(username=username, password=hashed_pw)
                db.session.add(user)
                db.session.commit()
                message = "Inscription réussie. Connectez-vous."
        elif action == "login":
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                return redirect('/account')
            else:
                message = "Identifiants incorrects."
        user = None 
    return render_template('account.html', message=message, user=user)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/account')

socketio = SocketIO(app)

@socketio.on('connect')
def handle_connect():
    print("Client connecté")

if __name__ == '__main__':
    socketio.run(app, debug=True)
    app.run(debug=True)
