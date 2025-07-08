# config.py
DB_USERNAME = 'postgres'
DB_PASSWORD = 'djamel10'
DB_HOST = 'localhost'
DB_NAME = 'db_binvision'
DB_PORT = '5432'

SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
SQLALCHEMY_TRACK_MODIFICATIONS = False
