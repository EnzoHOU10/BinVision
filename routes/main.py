from flask import Blueprint, render_template, request, redirect, session, send_from_directory
from models.models import TrashImage, ClassificationRule, Settings, User
from utils.classification import calculate_seuils_from_db
from utils.visualization import generate_matplotlib
from utils.decisiontree import create_tree, save_tree
from utils.imageprocessing import add_img
import pandas as pd
import os

google_maps_key = os.getenv("GOOGLE_MAPS_KEY")

main_bp = Blueprint('main', __name__)

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Servir les fichiers uploadés"""
    from app import app
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@main_bp.route('/', methods=['GET', 'POST'])
def home():
    """Page d'accueil avec carte des poubelles pleines"""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    rules = ClassificationRule.query.all()
    seuils, seuils_plein, seuils_vide = calculate_seuils_from_db()
    # Récupération des images de poubelles pleines pour la carte
    images = TrashImage.query.filter_by(annotation='Pleine').all()
    images_conv = []
    for img in images:
        images_conv.append({
            'id': img.id,
            'lat': img.lat,
            'lng': img.lng,
        })
    return render_template(
        'home.html', 
        rules=rules,
        seuils=seuils,
        seuils_plein=seuils_plein,
        seuils_vide=seuils_vide,
        images=images_conv,
        google_maps_key=google_maps_key,
        user=user
    )

@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Page de prédiction pour les utilisateurs"""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    settings = Settings.query.first()
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        if form_type == 'rules':
            # Mise à jour des règles de classification
            for name in request.form:
                if name not in ["use_auto_rules", "form_type"]:
                    try:
                        value = float(request.form[name])
                        from utils.classification import update_rule
                        update_rule(name, value)
                    except ValueError:
                        continue
            settings.use_auto_rules = 'use_auto_rules' in request.form
            from models.models import db
            db.session.commit()
            return redirect('/predict')
        elif form_type == 'upload':
            # Upload d'images
            images = request.files.getlist('image')
            for img in images:
                add_img(img)
            generate_matplotlib()
            return redirect('/predict')
    # Récupération des images classifiées automatiquement
    images = TrashImage.query.filter_by(type='Auto').all()
    images_conv = []
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
    return render_template('user.html', 
        rules=rules,
        seuils=seuils,
        seuils_plein=seuils_plein,
        seuils_vide=seuils_vide, 
        images=images_conv, 
        use_auto_rules=settings.use_auto_rules, 
        user=user)

@main_bp.route('/admin', methods=['GET', 'POST'])
def admin():
    """Page d'administration (entraînement du modèle)"""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    if user is None:
        return redirect('/account')
    if request.method == 'POST':
        # Upload d'images pour l'entraînement
        images = request.files.getlist('image')
        for img in images:
            add_img(img)
    # Entraînement du modèle avec les nouvelles données
    imgs = TrashImage.query.filter_by(type='Manual').all()
    if len(imgs)>1:
        try:
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
            save_tree(create_tree(X, y))
            generate_matplotlib()
        except ValueError as e:
            print(f"Erreur matplotlib : {e}")
    images = TrashImage.query.order_by(TrashImage.id.desc()).all()
    return render_template('settings.html', images=images, user=user)

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard avec visualisations et statistiques"""
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    rules = ClassificationRule.query.all()
    images = TrashImage.query.all()
    # Conversion des images pour le template
    images_conv = []
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
    # Génération ou récupération des graphiques
    files = ['totalgraph.png', 'pleinegraph.png', 'videgraph.png']
    files_exist = all(os.path.exists(os.path.join("static/matplotlib", f)) for f in files)
    if files_exist:
        paths = [f"../static/matplotlib/{f}" for f in files]
    else:
        if len(images)>0:
            paths = generate_matplotlib()
        else:
            paths = [f"../static/matplotlib/{f}" for f in files]
    return render_template(
        'dashboard.html', 
        rules=rules,
        images=images_conv,
        paths=paths,
        google_maps_key=google_maps_key,
        user=user
    )

@main_bp.route('/delete_marker/<int:marker_id>', methods=['DELETE'])
def delete_marker(marker_id):
    """Suppression d'un marqueur/image"""
    marker = TrashImage.query.get(marker_id)
    if marker:
        from models.models import db
        db.session.delete(marker)
        db.session.commit()
        return {'success': True, 'message': 'Marker supprimé'}, 200
    else:
        return {'success': False, 'message': 'Marker introuvable'}, 404