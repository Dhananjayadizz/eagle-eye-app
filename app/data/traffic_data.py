from app.extensions import db
from datetime import datetime

class TrafficData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    vehicle_count = db.Column(db.Integer, nullable=False)
    average_speed = db.Column(db.Float, nullable=False)
    congestion_level = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    lane_status = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'vehicle_count': self.vehicle_count,
            'average_speed': self.average_speed,
            'congestion_level': self.congestion_level,
            'location': self.location,
            'lane_status': self.lane_status
        } 