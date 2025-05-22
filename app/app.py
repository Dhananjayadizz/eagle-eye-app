from flask import Flask, render_template
from flask_socketio import SocketIO
from app.extensions import db, socketio
from app.routes.criticalEvents import critical_events_bp
from app.routes.traffic_analysis import traffic_analysis_bp
from app.routes.blockchain import blockchain_bp
from app.routes.pedestriansAnalysis import pedestrians_analysis_bp
from app.routes.liveStream import live_stream_bp
from app.core.gps_module import init_gps
import logging



from app.routes.criticalEvents import read_gps_data_from_serial, SERIAL_PORT_NAME, BAUDRATE
import threading
import os
import logging



# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    
    # Configure the app
    app.config["SECRET_KEY"] = "your-secret-key"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"  # This will create the database in the instance folder
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = "uploads"



    # UPLOAD_FOLDER = "uploads"
    # EXPORT_DIR = "exports"
    # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # os.makedirs(EXPORT_DIR, exist_ok=True)
    # app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    # app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
    # app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app, message_queue=None)
    
    # Register blueprints
    app.register_blueprint(critical_events_bp)
    app.register_blueprint(traffic_analysis_bp)
    app.register_blueprint(blockchain_bp, url_prefix='/blockchain')
    app.register_blueprint(pedestrians_analysis_bp)
    app.register_blueprint(live_stream_bp)
    
    # Initialize GPS
    init_gps()
    
    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/live')
    def live():
        return render_template('live_dashboard.html')

    @app.route('/section1')
    def section1():
        return render_template('critical_events_analysis.html')

    @app.route('/section2')
    def section2():
        return render_template('traffic_analysis.html')

    @app.route('/section3')
    def section3():
        return render_template('blockchain_store.html')

    @app.route('/section4')
    def section4():
        return render_template('pedestrians_analysis.html')

    return app


# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)




# Clear the exports directory on startup
# def clear_exports_directory():
#     try:
#         for filename in os.listdir(EXPORT_DIR):
#             file_path = os.path.join(EXPORT_DIR, filename)
#             if os.path.isfile(file_path) and filename.endswith('.xlsx'):
#                 os.unlink(file_path)  # Use os.unlink to remove the file
#         logger.info(f"Cleared contents of {EXPORT_DIR} on startup.")
#     except Exception as e:
#         logger.error(f"Error clearing exports directory on startup: {e}")


app = create_app()

if __name__ == '__main__':
    with app.app_context():
        # clear_exports_directory()
        db.create_all()
    socketio.run(app, debug=True) 