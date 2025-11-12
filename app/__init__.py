from flask import Flask
from flask_session import Session
from .api import api_bp, media_bp   # ⬅️ import BOTH blueprints
from .config import AppConfig
from werkzeug.middleware.proxy_fix import ProxyFix
import os

def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def create_app():
    # serve ./static so /static/wa-media/* works
    app = Flask(__name__, static_folder="static")
    app.config.from_object(AppConfig())

    # ---- Session (make sure dir exists) ----
    app.config['SECRET_KEY'] = app.config.get('SECRET_KEY') or os.urandom(24)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = app.config.get('SESSION_FILE_DIR') or './.flask_session/'
    _ensure_dir(app.config['SESSION_FILE_DIR'])
    Session(app)

    # ---- Reverse proxy headers (proto/host) ----
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    # ---- Ensure local media dir exists (where we copy captioned images) ----
    media_local_dir = os.getenv("MEDIA_LOCAL_DIR", "./static/wa-media")
    _ensure_dir(media_local_dir)

    # ---- Blueprints ----
    # API under /api/v1
    app.register_blueprint(api_bp, url_prefix="/api/v1")
    # Media EXACTLY at /static/wa-media/<file>
    app.register_blueprint(media_bp)  # no url_prefix override here

    # ---- Health & Docs ----
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/docs")
    def docs():
        return "<meta http-equiv='refresh' content='0; url=/api/v1/' />"

    return app
