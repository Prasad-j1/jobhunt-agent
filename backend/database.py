import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    basedir = os.path.abspath(os.path.dirname(__file__))
    db_path  = os.path.join(basedir, '..', 'jobhunter.db')

    app.config['SQLALCHEMY_DATABASE_URI']        = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    with app.app_context():
        from backend import models
        db.create_all()
        print("✓ Database ready")