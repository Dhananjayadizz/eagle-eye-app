from app.extensions import db
from datetime import datetime

class EventLog(db.Model):
    """Model for storing vehicle events"""
    __table_args__ = {'extend_existing': True}
    
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

    def to_dict(self):
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "ttc": self.ttc,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "motion_status": self.motion_status
        } 