from flask import Blueprint, render_template, jsonify, request
from app.extensions import db, socketio
from app.data.traffic_data import TrafficData
from datetime import datetime, timedelta
import logging

traffic_analysis_bp = Blueprint('traffic_analysis', __name__)
logger = logging.getLogger(__name__)

@traffic_analysis_bp.route('/section2')
def traffic_analysis():
    try:
        return render_template('traffic_analysis.html')
    except Exception as e:
        logging.error(f"Error rendering traffic analysis page: {str(e)}")
        return jsonify({"error": "Failed to load traffic analysis page"}), 500

@traffic_analysis_bp.route('/api/traffic', methods=['GET'])
def get_traffic_data():
    try:
        traffic_data = TrafficData.query.order_by(TrafficData.timestamp.desc()).all()
        return jsonify([data.to_dict() for data in traffic_data])
    except Exception as e:
        logger.error(f"Error fetching traffic data: {str(e)}")
        return jsonify({'error': 'Failed to fetch traffic data'}), 500

@traffic_analysis_bp.route('/api/traffic', methods=['POST'])
def create_traffic_data():
    try:
        data = request.get_json()
        traffic_data = TrafficData(
            vehicle_count=data['vehicle_count'],
            average_speed=data['average_speed'],
            congestion_level=data['congestion_level'],
            location=data['location'],
            lane_status=data['lane_status']
        )
        db.session.add(traffic_data)
        db.session.commit()
        socketio.emit('new_traffic_data', traffic_data.to_dict())
        return jsonify(traffic_data.to_dict()), 201
    except Exception as e:
        logger.error(f"Error creating traffic data: {str(e)}")
        return jsonify({'error': 'Failed to create traffic data'}), 500

@traffic_analysis_bp.route('/api/traffic/<int:data_id>', methods=['PUT'])
def update_traffic_data(data_id):
    try:
        traffic_data = TrafficData.query.get_or_404(data_id)
        data = request.get_json()
        
        traffic_data.vehicle_count = data.get('vehicle_count', traffic_data.vehicle_count)
        traffic_data.average_speed = data.get('average_speed', traffic_data.average_speed)
        traffic_data.congestion_level = data.get('congestion_level', traffic_data.congestion_level)
        traffic_data.location = data.get('location', traffic_data.location)
        traffic_data.lane_status = data.get('lane_status', traffic_data.lane_status)
        
        db.session.commit()
        socketio.emit('traffic_data_updated', traffic_data.to_dict())
        return jsonify(traffic_data.to_dict())
    except Exception as e:
        logger.error(f"Error updating traffic data: {str(e)}")
        return jsonify({'error': 'Failed to update traffic data'}), 500

@traffic_analysis_bp.route('/api/traffic/<int:data_id>', methods=['DELETE'])
def delete_traffic_data(data_id):
    try:
        traffic_data = TrafficData.query.get_or_404(data_id)
        db.session.delete(traffic_data)
        db.session.commit()
        socketio.emit('traffic_data_deleted', {'id': data_id})
        return '', 204
    except Exception as e:
        logger.error(f"Error deleting traffic data: {str(e)}")
        return jsonify({'error': 'Failed to delete traffic data'}), 500 