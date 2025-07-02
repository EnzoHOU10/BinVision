# config.py
DB_USERNAME = 'root'
DB_PASSWORD = '{En34{Ho64'
DB_HOST = 'localhost'
DB_NAME = 'db_binvision'

SQLALCHEMY_DATABASE_URI = f'mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
SQLALCHEMY_TRACK_MODIFICATIONS = False