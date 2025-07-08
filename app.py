from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from models.models import db, ClassificationRule, Settings, User, initialize_rules, create_admin_account
from routes.auth import auth_bp
from routes.main import main_bp
from werkzeug.security import generate_password_hash
from utils.testing import test_manual, test_sklearn
from flask import send_from_directory
from dotenv import load_dotenv

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Initialisation de l'application Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "projetbinvision2025"
# Initialisation des extensions
socketio = SocketIO(app)
# Initialisation des clés
load_dotenv()
# Enregistrement des blueprints
app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)
# Initialisation de l'application
def initializeapp():
    db.init_app(app)
    with app.app_context():
        db.create_all()
        initialize_rules()
        create_admin_account()

@socketio.on('connect')
def handle_connect():
    print("Client connecté")

if __name__ == '__main__':
    initializeapp()
    with app.app_context():
        try:
            test_manual()
            test_sklearn()
        except Exception as e:
            print(f"Erreur lors des tests: {e}")
    socketio.run(app, debug=True)
    app.run(debug=True)