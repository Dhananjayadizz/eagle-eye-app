from app.extensions import db
from datetime import datetime

class PedestrianData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    pedestrian_count = db.Column(db.Integer, nullable=False)
    crossing_status = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    risk_level = db.Column(db.String(20), default='low')

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'pedestrian_count': self.pedestrian_count,
            'crossing_status': self.crossing_status,
            'location': self.location,
            'risk_level': self.risk_level
        } 