from app.extensions import db
from datetime import datetime
import json

class PedestrianData(db.Model):
    __tablename__ = 'pedestrian_data'
    
    id = db.Column(db.Integer, primary_key=True)
    pedestrian_id = db.Column(db.String(50), nullable=False)
    intent_score = db.Column(db.Float, nullable=False)
    speed = db.Column(db.Float, nullable=False)
    location = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __init__(self, pedestrian_id, intent_score, speed, location, timestamp=None):
        self.pedestrian_id = pedestrian_id
        self.intent_score = intent_score
        self.speed = speed
        self.location = location
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self):
        return {
            'id': self.id,
            'pedestrian_id': self.pedestrian_id,
            'intent_score': self.intent_score,
            'speed': self.speed,
            'location': self.location,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __repr__(self):
        return f'<PedestrianData {self.pedestrian_id}>' 