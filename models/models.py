from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db

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
    """Crée le compte administrateur par défaut"""
    admin_username = "admin"
    admin_email = "admin@binvision.com"
    admin_password = "12345"
    admin_role = "admin"
    if not User.query.filter_by(email=admin_email).first():
        hashed_pw = generate_password_hash(admin_password)
        admin = User(username=admin_username, email=admin_email, password=hashed_pw, role=admin_role)
        db.session.add(admin)
        db.session.commit()

def initialize_rules():
    """Initialise les règles de classification par défaut"""
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

def get_rule(name, default=0):
    """Récupère une règle de classification"""
    rule = ClassificationRule.query.filter_by(name=name).first()
    return rule.value if rule else default

def update_rule(name, value):
    """Met à jour une règle de classification"""
    rule = ClassificationRule.query.filter_by(name=name).first()
    if rule:
        rule.value = value
    else:
        rule = ClassificationRule(name=name, value=value)
        db.session.add(rule)
    db.session.commit()