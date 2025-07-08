# config.py
DB_USERNAME = 'user'
DB_PASSWORD = 'IbgTIo3ThIXDLc0dywDcuvAEcznsv576'
DB_HOST = 'dpg-d1mg24h5pdvs73cvv2s0-a.frankfurt-postgres.render.com'
DB_NAME = 'db_binvision'
DB_PORT = '5432'

SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
SQLALCHEMY_TRACK_MODIFICATIONS = False
