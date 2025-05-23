from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import threading
import os
import logging
import numpy as np
import pandas as pd
import torch
import joblib
import math
import time
from flask_socketio import emit
import base64
from datetime import datetime

# Import from the project structure - updated to match your actual directory structure
from app.core.sort import Sort
from app.core.vehicle_tracker import VehicleTracker
from app.core.motion_detection import detect_motion_changes
from app.core.utils import (
    calculate_ttc, estimate_frontier_speed, detect_lanes, 
    get_ego_lane_bounds, draw_lanes, calculate_iou, 
    haversine_distance, calculate_speed_from_gps, draw_speedometer
)
from app.core.gps_module import get_gps_data

# Import the socketio instance from extensions instead of main
from app.extensions import socketio

# Create blueprint
live_stream_bp = Blueprint('live_stream', __name__)
logger = logging.getLogger(__name__)

# Global variables
cap = None
processing_thread = None
processing_settings = {
    'quality': 'medium',
    'fps': 10,
    'frame_skip': 2,  # Process every nth frame
    'confidence': 0.4,  # Lower confidence threshold for faster detection
    'resolution': (640, 480)
}

# Initialize models
try:
    model = None
    tracker = Sort()
    kalman_tracker = VehicleTracker()
    
    # These will be initialized when needed
    anomaly_model = None
    scaler = None
    frontier_clf = None
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
except Exception as e:
    logger.error(f"Error initializing models: {e}")

def load_models():
    """Load all required models"""
    global model, anomaly_model, scaler, frontier_clf
    
    try:
        # Load YOLO model if not already loaded
        if model is None:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # Use the smallest model for speed
            model.to(device)
            logger.info("YOLO model loaded successfully")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # These models are optional and will be loaded if available
        try:
            anomaly_model_path = os.path.join(models_dir, 'frontier_anomaly_model.pkl')
            if os.path.exists(anomaly_model_path):
                anomaly_model = joblib.load(anomaly_model_path)
                logger.info("Anomaly model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load anomaly model: {e}")
            anomaly_model = None
            
        try:
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}")
            scaler = None
            
        try:
            frontier_clf_path = os.path.join(models_dir, 'frontier_classifier.pkl')
            if os.path.exists(frontier_clf_path):
                with open(frontier_clf_path, "rb") as f:
                    frontier_clf = joblib.load(f)
                logger.info("Frontier classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load frontier classifier: {e}")
            frontier_clf = None
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False
        
    return True

def process_live_stream(camera_id):
    """Process live video stream"""
    # Ensure models are loaded
    if not load_models():
        logger.error("Failed to load required models")
        socketio.emit('stream_error', {'error': 'Failed to load required models'})
        return
        
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera: {camera_id}")
        socketio.emit('stream_error', {'error': f'Failed to open camera with index: {camera_id}'})
        return

    # Initialize variables
    FPS = processing_settings['fps']
    FRAME_TIME = 1 / FPS
    prev_frame = None
    frame_count = 0
    prev_tracks = {}
    ego_gps_history = {}
    vehicle_history = {}
    frame_skip = processing_settings['frame_skip']
    confidence_threshold = processing_settings['confidence']
    target_resolution = processing_settings['resolution']

    logging.info(f"Camera opened: FPS={FPS}, FRAME_TIME={FRAME_TIME}, frame_skip={frame_skip}, resolution={target_resolution}")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame from camera")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:  # Process every nth frame
                continue

            # Resize frame to target resolution for faster processing
            frame = cv2.resize(frame, target_resolution)
            height, width, _ = frame.shape

            # Lane detection - simplified for speed
            lane_lines = detect_lanes(frame)
            left_lane_x, right_lane_x, _, _ = get_ego_lane_bounds(lane_lines, width, height)
            
            # Skip drawing lanes for performance
            # draw_lanes(frame, lane_lines)

            # Get GPS data
            current_gps = get_gps_data()
            ego_speed = current_gps.get("speed", 0.0)
            
            # Simplified motion detection
            motion_status = "Normal Motion"
            prev_frame = frame.copy()

            # Calculate ego vehicle speed using GPS data
            lat, lon = current_gps.get("latitude", 0.0), current_gps.get("longitude", 0.0)
            ego_speed_gps = calculate_speed_from_gps(ego_gps_history, lat, lon, frame_count, FRAME_TIME)

            # Draw speedometer
            draw_speedometer(frame, ego_speed_gps)

            # Object detection and tracking - optimized
            results = model(frame, verbose=False, conf=confidence_threshold)[0]
            detections = []
            
            # Only process vehicle classes (2, 3, 5, 7) for speed
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                if int(class_id) in [2, 3, 5, 7]:  # COCO classes for vehicles
                    detections.append([x1, y1, x2, y2, score])

            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 5)))
            frontier_vehicle = None
            min_distance = float('inf')

            for track in tracked_objects:
                if len(track) < 5:
                    continue
                
                # Ensure track has at least 5 elements before unpacking
                if len(track) >= 5:
                    x1, y1, x2, y2, track_id = map(int, track[:5])

                    color = (255, 0, 0)
                    event_type = "Tracked"
                    ttc = None
                    vehicle_motion = "Normal Motion"

                    # Calculate distance to ego vehicle (simplified - distance from bottom of frame)
                    distance = height - y2 # This is pixel distance, not real-world distance

                    # Find the closest vehicle (potential frontier vehicle)
                    if distance < min_distance and int(track_id) != -1:
                        min_distance = distance
                        frontier_vehicle = track

                    # Check for critical events related to the frontier vehicle
                    is_critical_event = False
                    if frontier_vehicle is not None and np.array_equal(track[:5], frontier_vehicle[:5]):
                        color = (0, 255, 0) # Green for frontier vehicle
                        event_type = "Frontier"
                        y_center = (y1 + y2) // 2

                        # Estimate frontier vehicle speed
                        frontier_speed_px = estimate_frontier_speed(track_id, y_center, frame_count, FRAME_TIME, vehicle_history)
                        PIXELS_PER_METER_EST = 0.1
                        frontier_speed_kmh = (frontier_speed_px * PIXELS_PER_METER_EST) * 3.6

                        # Calculate TTC
                        distance_meters = distance * PIXELS_PER_METER_EST
                        if ego_speed is not None:
                            ttc = calculate_ttc(ego_speed, frontier_speed_kmh, distance_meters)
                        else:
                            ttc = float('inf')

                        if ttc is not None and ttc != float('inf') and ttc < 2.0:
                            event_type = "Near Collision"
                            is_critical_event = True

                        # Predict next position using Kalman filter
                        x_center = (x1 + x2) // 2
                        pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
                        cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

                        # Skip anomaly detection for performance
                        # if scaler is not None and anomaly_model is not None:
                        #     try:
                        #         features = pd.DataFrame([[frontier_speed_kmh, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
                        #         scaled_features_array = scaler.transform(features)
                        #         scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
                        #         if anomaly_model.predict(scaled_features)[0] == -1:
                        #             event_type = f"{event_type} - Anomaly"
                        #             is_critical_event = True
                        #     except Exception as e:
                        #         logging.error(f"Error during anomaly detection: {e}")

                        # Motion check - simplified
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if track_id in prev_tracks:
                            prev_cx, prev_cy = prev_tracks[track_id]
                            displacement_px = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                            if displacement_px < 5 and frontier_speed_kmh < 10:
                                vehicle_motion = "Sudden Stop Detected!"
                                is_critical_event = True
                            elif displacement_px > 10 and frontier_speed_kmh > 50:
                                vehicle_motion = "Harsh Braking"
                                is_critical_event = True
                        prev_tracks[track_id] = (cx, cy)

                        # Collision detection - simplified
                        current_bbox = [x1, y1, x2, y2]
                        for other_track in tracked_objects:
                            if len(other_track) >= 5 and int(other_track[4]) != track_id:
                                other_bbox = [int(other_track[0]), int(other_track[1]), int(other_track[2]), int(other_track[3])]
                                if calculate_iou(current_bbox, other_bbox) > 0.2:
                                    vehicle_motion = "Collided"
                                    is_critical_event = True
                                    break

                    # Draw bounding box and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add labels - simplified for performance
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    
                    # Only show ID and status for better performance
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    
                    # Only show additional info for frontier vehicle
                    if frontier_vehicle is not None and np.array_equal(track[:5], frontier_vehicle[:5]):
                        cv2.putText(frame, f"Motion: {vehicle_motion}", (x1, y1-30), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        
                        if 'frontier_speed_kmh' in locals():
                            cv2.putText(frame, f"Speed: {frontier_speed_kmh:.1f} km/h", (x1, y1-50), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        
                        if ttc is not None:
                            ttc_text = f"TTC: {ttc:.1f}s" if ttc != float('inf') else "TTC: N/A"
                            cv2.putText(frame, ttc_text, (x1, y1-70), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                    # Emit critical events
                    if is_critical_event:
                        event_data = {
                            "event_type": event_type,
                            "timestamp": datetime.utcnow().isoformat(),
                            "ttc": "N/A" if ttc is None or ttc == float("inf") else round(ttc, 2),
                            "latitude": current_gps.get("latitude"),
                            "longitude": current_gps.get("longitude"),
                            "vehicle_id": int(track_id),
                            "motion_status": vehicle_motion,
                            "is_critical": is_critical_event
                        }
                        try:
                            socketio.emit('new_event', event_data)
                        except Exception as e:
                            logging.error(f"Error emitting new event via socketio: {e}")

            # Encode and send frame via SocketIO - optimized quality
            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if success:
                frame_bytes = buffer.tobytes()
                try:
                    # Convert to base64 for proper transmission via SocketIO
                    b64_frame = base64.b64encode(frame_bytes).decode('utf-8')
                    b64_data = 'data:image/jpeg;base64,' + b64_frame
                    socketio.emit('processed_frame', b64_data)
                except Exception as e:
                    logging.error(f"Error emitting frame via socketio: {e}")

    except Exception as e:
        logging.error(f"Error in process_live_stream: {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        logging.info("Live stream processing stopped.")
        # Don't use db.session.remove() if db is not defined

@socketio.on('client_frame')
def handle_client_frame(base64_img):
    """Process frames sent from client"""
    try:
        # Ensure models are loaded
        if not load_models():
            logger.error("Failed to load required models")
            return
            
        # Decode base64 image
        header, encoded = base64_img.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Failed to decode frame")
            return

        # Process frame - use target resolution
        frame = cv2.resize(frame, processing_settings['resolution'])
        
        # Object detection and tracking - optimized
        results = model(frame, verbose=False, conf=processing_settings['confidence'])[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > processing_settings['confidence'] and int(class_id) in [2, 3, 5, 7]:  # Vehicle classes
                detections.append([x1, y1, x2, y2, score])
        
        tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 5)))
        
        # Draw bounding boxes and labels - simplified
        for track in tracked_objects:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode to JPEG and base64 - optimized quality
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            logger.error("Failed to encode frame")
            return

        b64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        b64_data = 'data:image/jpeg;base64,' + b64_frame

        emit('processed_frame', b64_data)
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

@socketio.on('start_live_processing')
def handle_start_live_processing(data):
    """Handle start of live processing"""
    logger.info(f"Received start_live_processing signal with data: {data}")
    
    # Get camera index
    camera_index = data.get('cameraIndex')
    if camera_index is None:
        logger.error("No cameraIndex provided in start_live_processing signal")
        socketio.emit('stream_error', {'error': 'No camera index provided.'})
        return

    # Update processing settings from client
    if 'quality' in data:
        quality = data.get('quality', 'medium')
        if quality == 'low':
            processing_settings['resolution'] = (320, 240)
            processing_settings['frame_skip'] = 3
            processing_settings['confidence'] = 0.3
        elif quality == 'medium':
            processing_settings['resolution'] = (640, 480)
            processing_settings['frame_skip'] = 2
            processing_settings['confidence'] = 0.4
        elif quality == 'high':
            processing_settings['resolution'] = (1280, 720)
            processing_settings['frame_skip'] = 1
            processing_settings['confidence'] = 0.5
    
    if 'fps' in data:
        try:
            fps = int(data.get('fps', 10))
            processing_settings['fps'] = fps
        except (ValueError, TypeError):
            pass

    # Convert to integer if possible
    try:
        camera_index = int(camera_index)
    except ValueError:
        # If it's not an integer, assume it's a device ID string
        pass

    # Start processing thread
    global processing_thread
    if processing_thread is None or not processing_thread.is_alive():
        logger.info(f"Starting live stream processing thread for camera: {camera_index}")
        processing_thread = threading.Thread(target=process_live_stream, args=(camera_index,), daemon=True)
        processing_thread.start()
        socketio.emit('status', {'message': f'Started processing for camera {camera_index} with settings: {processing_settings}'})

@socketio.on('stop_live_processing')
def handle_stop_live_processing():
    """Handle stop of live processing"""
    global cap, processing_thread
    logging.info("Received stop_live_processing request.")
    
    # Stop the camera capture
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    
    # The processing thread should exit when cap is released
    processing_thread = None
    
    socketio.emit('status', {'message': 'Stopped live video processing'})

@live_stream_bp.route('/live')
def live_dashboard():
    """Render the live dashboard page"""
    return render_template('live_dashboard.html')
