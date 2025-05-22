from app.extensions import db
from datetime import datetime
import numpy as np

class EventLog(db.Model):
    __tablename__ = 'event_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, nullable=True)
    event_type = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    x1 = db.Column(db.Float, nullable=True)
    y1 = db.Column(db.Float, nullable=True)
    x2 = db.Column(db.Float, nullable=True)
    y2 = db.Column(db.Float, nullable=True)
    ttc = db.Column(db.Float, nullable=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    motion_status = db.Column(db.String(100), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'vehicle_id': int(self.vehicle_id) if self.vehicle_id is not None else None,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'x1': float(self.x1) if self.x1 is not None else None,
            'y1': float(self.y1) if self.y1 is not None else None,
            'x2': float(self.x2) if self.x2 is not None else None,
            'y2': float(self.y2) if self.y2 is not None else None,
            'ttc': float(self.ttc) if self.ttc is not None else None,
            'latitude': float(self.latitude) if self.latitude is not None else None,
            'longitude': float(self.longitude) if self.longitude is not None else None,
            'motion_status': self.motion_status
        } 