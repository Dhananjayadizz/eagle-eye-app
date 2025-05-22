from flask import Blueprint, render_template, jsonify, request, current_app
from app.extensions import db, socketio
from app.data.pedestrian_data import PedestrianData
from datetime import datetime
import logging
import os
import cv2
from ultralytics import YOLO
import threading

pedestrians_analysis_bp = Blueprint('pedestrians_analysis', __name__)
logger = logging.getLogger(__name__)

# Initialize YOLO model for pedestrian detection
model = YOLO('yolov8n.pt')

@pedestrians_analysis_bp.route('/section4')
def pedestrians_analysis():
    try:
        return render_template('pedestrians_analysis.html')
    except Exception as e:
        logging.error(f"Error rendering pedestrians analysis page: {str(e)}")
        return jsonify({"error": "Failed to load pedestrians analysis page"}), 500

@pedestrians_analysis_bp.route('/upload_pedestrian_video', methods=['POST'])
def upload_pedestrian_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the video file
        video_path = os.path.join(upload_dir, 'pedestrian_video.mp4')
        video_file.save(video_path)
        
        # Start video processing in a background thread
        socketio.start_background_task(process_pedestrian_video, video_path)
        
        return jsonify({
            'success': True,
            'video_url': '/uploads/pedestrian_video.mp4'
        })
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_pedestrian_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 == 0:  # Process every other frame
                continue

            # Resize frame for processing
            frame = cv2.resize(frame, (640, 480))
            
            # Perform object detection
            results = model(frame)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Class 0 is person in COCO dataset
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        if conf > 0.5:  # Confidence threshold
                            # Calculate pedestrian metrics
                            intent_score = calculate_intent_score(x1, y1, x2, y2, frame.shape)
                            speed = calculate_pedestrian_speed(x1, y1, x2, y2)
                            
                            # Create pedestrian event
                            event = {
                                'id': frame_count,
                                'timestamp': datetime.now().isoformat(),
                                'pedestrian_id': f'P{frame_count}',
                                'intent_score': intent_score,
                                'speed': speed,
                                'location': {'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2}
                            }
                            
                            # Emit event through socket
                            socketio.emit('pedestrian_event', event)
                            
                            # Save to database
                            pedestrian_data = PedestrianData(
                                pedestrian_id=event['pedestrian_id'],
                                intent_score=event['intent_score'],
                                speed=event['speed'],
                                location=event['location'],
                                timestamp=datetime.now()
                            )
                            db.session.add(pedestrian_data)
            
            # Commit to database every 30 frames
            if frame_count % 30 == 0:
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Commit failed: {e}")
        
        # Emit completion event
        socketio.emit('analysis_complete')
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

def calculate_intent_score(x1, y1, x2, y2, frame_shape):
    """Calculate pedestrian intent score based on position and movement"""
    height, width = frame_shape[:2]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate distance from center of frame
    distance_from_center = ((center_x - width/2)**2 + (center_y - height/2)**2)**0.5
    max_distance = ((width/2)**2 + (height/2)**2)**0.5
    
    # Normalize distance to 0-1 range (closer to center = higher score)
    distance_score = 1 - (distance_from_center / max_distance)
    
    # Calculate size score (larger = closer = higher score)
    size = (x2 - x1) * (y2 - y1)
    max_size = width * height
    size_score = min(size / max_size * 10, 1)  # Cap at 1
    
    # Combine scores (weighted average)
    intent_score = (distance_score * 0.6 + size_score * 0.4)
    
    return min(max(intent_score, 0), 1)  # Ensure score is between 0 and 1

def calculate_pedestrian_speed(x1, y1, x2, y2):
    """Calculate pedestrian speed in pixels per frame"""
    # This is a simplified calculation. In a real system, you would track
    # pedestrians across frames and calculate actual speed
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

@pedestrians_analysis_bp.route('/export_pedestrian_events')
def export_pedestrian_events():
    try:
        # Get all pedestrian events
        events = PedestrianData.query.order_by(PedestrianData.timestamp.desc()).all()
        
        # Create CSV content
        csv_content = "ID,Timestamp,Pedestrian ID,Intent Score,Speed,Location\n"
        for event in events:
            csv_content += f"{event.id},{event.timestamp},{event.pedestrian_id},{event.intent_score},{event.speed},{event.location}\n"
        
        # Save to file
        export_dir = os.path.join(current_app.root_path, '..', 'exports')
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f'pedestrian_events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        with open(export_path, 'w') as f:
            f.write(csv_content)
        
        return jsonify({'success': True, 'file_path': export_path})
    except Exception as e:
        logger.error(f"Error exporting pedestrian events: {str(e)}")
        return jsonify({'error': str(e)}), 500

@pedestrians_analysis_bp.route('/clear_pedestrian_files', methods=['POST'])
def clear_pedestrian_files():
    try:
        # Clear database entries
        PedestrianData.query.delete()
        db.session.commit()
        
        # Clear video file
        video_path = os.path.join(current_app.root_path, '..', 'uploads', 'pedestrian_video.mp4')
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing pedestrian files: {str(e)}")
        return jsonify({'error': str(e)}), 500 