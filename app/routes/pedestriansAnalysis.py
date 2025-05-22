from flask import Blueprint, render_template, jsonify, request
from app.extensions import db, socketio
from app.data.pedestrian_data import PedestrianData
from datetime import datetime, timedelta
import logging

pedestrians_analysis_bp = Blueprint('pedestrians_analysis', __name__)
logger = logging.getLogger(__name__)

@pedestrians_analysis_bp.route('/section4')
def pedestrians_analysis():
    try:
        return render_template('pedestrians_analysis.html')
    except Exception as e:
        logging.error(f"Error rendering pedestrians analysis page: {str(e)}")
        return jsonify({"error": "Failed to load pedestrians analysis page"}), 500

@pedestrians_analysis_bp.route('/api/pedestrians', methods=['GET'])
def get_pedestrian_data():
    try:
        pedestrian_data = PedestrianData.query.order_by(PedestrianData.timestamp.desc()).all()
        return jsonify([data.to_dict() for data in pedestrian_data])
    except Exception as e:
        logger.error(f"Error fetching pedestrian data: {str(e)}")
        return jsonify({'error': 'Failed to fetch pedestrian data'}), 500

@pedestrians_analysis_bp.route('/api/pedestrians', methods=['POST'])
def create_pedestrian_data():
    try:
        data = request.get_json()
        pedestrian_data = PedestrianData(
            pedestrian_count=data['pedestrian_count'],
            crossing_status=data['crossing_status'],
            location=data['location'],
            risk_level=data.get('risk_level', 'low')
        )
        db.session.add(pedestrian_data)
        db.session.commit()
        socketio.emit('new_pedestrian_data', pedestrian_data.to_dict())
        return jsonify(pedestrian_data.to_dict()), 201
    except Exception as e:
        logger.error(f"Error creating pedestrian data: {str(e)}")
        return jsonify({'error': 'Failed to create pedestrian data'}), 500

@pedestrians_analysis_bp.route('/api/pedestrians/<int:data_id>', methods=['PUT'])
def update_pedestrian_data(data_id):
    try:
        pedestrian_data = PedestrianData.query.get_or_404(data_id)
        data = request.get_json()
        
        pedestrian_data.pedestrian_count = data.get('pedestrian_count', pedestrian_data.pedestrian_count)
        pedestrian_data.crossing_status = data.get('crossing_status', pedestrian_data.crossing_status)
        pedestrian_data.location = data.get('location', pedestrian_data.location)
        pedestrian_data.risk_level = data.get('risk_level', pedestrian_data.risk_level)
        
        db.session.commit()
        socketio.emit('pedestrian_data_updated', pedestrian_data.to_dict())
        return jsonify(pedestrian_data.to_dict())
    except Exception as e:
        logger.error(f"Error updating pedestrian data: {str(e)}")
        return jsonify({'error': 'Failed to update pedestrian data'}), 500

@pedestrians_analysis_bp.route('/api/pedestrians/<int:data_id>', methods=['DELETE'])
def delete_pedestrian_data(data_id):
    try:
        pedestrian_data = PedestrianData.query.get_or_404(data_id)
        db.session.delete(pedestrian_data)
        db.session.commit()
        socketio.emit('pedestrian_data_deleted', {'id': data_id})
        return '', 204
    except Exception as e:
        logger.error(f"Error deleting pedestrian data: {str(e)}")
        return jsonify({'error': 'Failed to delete pedestrian data'}), 500 