from flask import Blueprint, render_template, jsonify, request, current_app, Response
from app.extensions import db, socketio
from app.data.pedestrian_data import PedestrianData
from datetime import datetime
import logging
import os
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import io
import threading
import time
from threading import Lock

pedestrians_analysis_bp = Blueprint('pedestrians_analysis', __name__)
logger = logging.getLogger(__name__)

# Initialize YOLO model for pedestrian detection
model = YOLO('yolov8n.pt')

# Global variables
pedestrian_history = {}
pedestrian_lock = Lock()
current_frame = None  # Initialize global current_frame

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
        
        # Validate file type
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return jsonify({'error': 'Please select a valid video file'}), 400
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the video file
        video_path = os.path.join(upload_dir, 'pedestrian_video.mp4')
        video_file.save(video_path)
        
        # Generate a unique video ID
        video_id = f"pedestrian_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize a default frame to avoid None in video feed
        global current_frame
        try:
            # Try to read the first frame to use as initial frame
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    with pedestrian_lock:
                        current_frame = frame.copy()
                cap.release()
        except Exception as e:
            logger.error(f"Error initializing first frame: {str(e)}")
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'video_url': '/uploads/pedestrian_video.mp4'
        })
    except Exception as e:
        logger.error(f"Error uploading pedestrian video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('start_pedestrian_analysis')
def handle_start_pedestrian_analysis(data):
    """Handle start pedestrian analysis event from client"""
    try:
        video_id = data.get('video_id')
        logger.info(f"Starting pedestrian analysis for video ID: {video_id}")
        
        # Get the app instance from the current request context
        app = current_app._get_current_object()
        
        # Start video processing in a background thread with app context
        socketio.start_background_task(process_pedestrian_video_with_context, app)
        
    except Exception as e:
        logger.error(f"Error starting pedestrian analysis: {str(e)}")
        socketio.emit('error', {'message': f"Error starting analysis: {str(e)}"})

def process_pedestrian_video_with_context(app):
    """Wrap the video processing function with application context"""
    with app.app_context():
        video_path = os.path.join(app.root_path, '..', 'uploads', 'pedestrian_video.mp4')
        process_pedestrian_video(video_path)

def process_pedestrian_video(video_path):
    """Process the pedestrian video and emit events"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            socketio.emit('error', {'message': f"Failed to open video: {video_path}"})
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1.0 / fps if fps > 0 else 0.033  # Default to 30fps if not available
        
        frame_count = 0
        pedestrian_id_counter = 1
        
        # Create a video writer for the processed output
        output_path = os.path.join(os.path.dirname(video_path), 'processed_pedestrian_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Use global current_frame
        global current_frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:  # Process every third frame for performance
                continue

            # Store current frame for video feed - always update even if not processing
            with pedestrian_lock:
                current_frame = frame.copy()
            
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
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Calculate pedestrian metrics
                            intent_score = calculate_intent_score(x1, y1, x2, y2, frame.shape)
                            speed = calculate_pedestrian_speed(x1, y1, x2, y2, frame_count, frame_time)
                            
                            # Determine color based on intent score
                            if intent_score >= 0.7:
                                color = (0, 0, 255)  # Red for high risk
                                status = "HIGH RISK"
                            elif intent_score >= 0.5:
                                color = (0, 165, 255)  # Orange for medium risk
                                status = "MEDIUM RISK"
                            else:
                                color = (0, 255, 0)  # Green for low risk
                                status = "LOW RISK"
                            
                            # Add text labels
                            cv2.putText(frame, f"ID: P{pedestrian_id_counter}", (x1, y1 - 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, f"Risk: {int(intent_score * 100)}%", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(frame, status, (x1, y2 + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Create pedestrian event
                            event = {
                                'timestamp': datetime.now().isoformat(),
                                'pedestrian_id': f'P{pedestrian_id_counter}',
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
                                location=str(event['location']),
                                timestamp=datetime.now()
                            )
                            db.session.add(pedestrian_data)
                            
                            pedestrian_id_counter += 1
            
            # Write the processed frame
            out.write(frame)
            
            # Update the current_frame with the processed frame (with bounding boxes)
            with pedestrian_lock:
                current_frame = frame.copy()
            
            # Commit to database every 30 frames
            if frame_count % 30 == 0:
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Database commit failed: {e}")
                    db.session.rollback()
        
        # Emit completion event
        socketio.emit('analysis_complete')
        
        # Release resources
        out.release()
        
    except Exception as e:
        logger.error(f"Error processing pedestrian video: {str(e)}")
        socketio.emit('error', {'message': f"Error processing video: {str(e)}"})
    finally:
        if cap is not None:
            cap.release()

@pedestrians_analysis_bp.route('/pedestrian_video_feed')
def pedestrian_video_feed():
    """Stream the processed video feed"""
    def generate():
        global current_frame
        while True:
            with pedestrian_lock:
                if current_frame is not None:
                    # Encode the frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if not ret:
                        continue
                    
                    # Convert to bytes and yield
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # If no frame is available, yield a blank frame
                    blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(blank_frame, "Waiting for video...", (150, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    ret, buffer = cv2.imencode('.jpg', blank_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Sleep to control frame rate
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

def calculate_pedestrian_speed(x1, y1, x2, y2, frame_count, frame_time):
    """Calculate pedestrian speed in pixels per frame with tracking"""
    pedestrian_id = f"{x1}_{y1}_{x2}_{y2}"  # Simple ID based on position
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    if pedestrian_id not in pedestrian_history:
        pedestrian_history[pedestrian_id] = {
            "last_x": center_x,
            "last_y": center_y,
            "last_frame": frame_count,
            "speed": 0.0
        }
        return 0.0
    
    # Calculate time difference
    last_frame = pedestrian_history[pedestrian_id]["last_frame"]
    time_diff = (frame_count - last_frame) * frame_time
    
    if time_diff <= 0:
        return pedestrian_history[pedestrian_id]["speed"]
    
    # Calculate displacement
    last_x = pedestrian_history[pedestrian_id]["last_x"]
    last_y = pedestrian_history[pedestrian_id]["last_y"]
    displacement = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5
    
    # Calculate speed
    speed = displacement / time_diff
    
    # Smooth speed with exponential moving average
    alpha = 0.7
    smoothed_speed = alpha * speed + (1 - alpha) * pedestrian_history[pedestrian_id]["speed"]
    
    # Update history
    pedestrian_history[pedestrian_id]["last_x"] = center_x
    pedestrian_history[pedestrian_id]["last_y"] = center_y
    pedestrian_history[pedestrian_id]["last_frame"] = frame_count
    pedestrian_history[pedestrian_id]["speed"] = smoothed_speed
    
    return smoothed_speed

@pedestrians_analysis_bp.route('/export_pedestrian_events')
def export_pedestrian_events():
    try:
        # Get all pedestrian events
        events = PedestrianData.query.order_by(PedestrianData.timestamp.desc()).all()
        
        # Create Excel file in memory
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        
        # Convert to DataFrame
        data = []
        for event in events:
            data.append({
                'ID': event.id,
                'Timestamp': event.timestamp,
                'Pedestrian ID': event.pedestrian_id,
                'Intent Score': f"{event.intent_score:.2f}",
                'Speed': f"{event.speed:.2f}",
                'Location': event.location
            })
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Pedestrian Events', index=False)
        
        # Save the Excel file
        writer.close()
        output.seek(0)
        
        # Generate filename with timestamp
        filename = f'pedestrian_events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        # Return the file as attachment
        return Response(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
    except Exception as e:
        logger.error(f"Error exporting pedestrian events: {str(e)}")
        return jsonify({'error': str(e)}), 500

@pedestrians_analysis_bp.route('/clear_pedestrian_files', methods=['POST'])
def clear_pedestrian_files():
    try:
        # Clear database entries
        PedestrianData.query.delete()
        db.session.commit()
        
        # Clear video files
        upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
        video_path = os.path.join(upload_dir, 'pedestrian_video.mp4')
        processed_path = os.path.join(upload_dir, 'processed_pedestrian_video.mp4')
        
        if os.path.exists(video_path):
            os.remove(video_path)
        
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing pedestrian files: {str(e)}")
        return jsonify({'error': str(e)}), 500
