from app.app import create_app
from app.extensions import db
import os

def init_database():
    """Initialize the database in the instance folder"""
    app = create_app()
    
    # Ensure instance folder exists
    instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
        print(f"Created instance folder at {instance_path}")
    
    with app.app_context():
        # Create all database tables
        db.create_all()
        print("Database tables created successfully in instance folder")

if __name__ == '__main__':
    init_database() 