from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()
socketio = SocketIO() 

class EventLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, nullable=False)
    event_type = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    x1 = db.Column(db.Integer)
    y1 = db.Column(db.Integer)
    x2 = db.Column(db.Integer)
    y2 = db.Column(db.Integer)
    ttc = db.Column(db.Float, nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    motion_status = db.Column(db.String(50), nullable=True)