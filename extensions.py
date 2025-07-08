from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sqlalchemy import create_engine
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS

engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
db = SQLAlchemy(session_options={"bind": engine})
socketio = SocketIO()