from flask import Flask
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from dotenv import load_dotenv
from extensions import db, socketio
from models.models import ClassificationRule, Settings, User, initialize_rules, create_admin_account
from routes.auth import auth_bp
from routes.main import main_bp
from utils.testing import test_manual, test_sklearn

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = "projetbinvision2025"

    load_dotenv()

    db.init_app(app)
    socketio.init_app(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)

    with app.app_context():
        db.create_all()
        initialize_rules()
        create_admin_account()
        try:
            test_manual()
            test_sklearn()
        except Exception as e:
            print(f"Erreur lors des tests: {e}")

    return app
