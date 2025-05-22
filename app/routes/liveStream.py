# from flask import Blueprint, render_template, Response, jsonify, request
# import cv2
# import threading
# import os
# import logging
# import numpy as np
# import pandas as pd
# import torch
# import joblib
# import math
# import time
# from app.extensions import db, socketio
# from app.core.sort import Sort
# from app.core.vehicle_tracker import VehicleTracker
# from app.core.motion_detection import detect_motion_changes
# from app.core.utils import calculate_ttc, estimate_frontier_speed, detect_lanes, get_ego_lane_bounds, draw_lanes, calculate_iou, haversine_distance, calculate_speed_from_gps, draw_speedometer
# from app.core.gps_module import get_gps_data # Assuming get_gps_data is now in gps_module
# import pickle
# from datetime import datetime
# from ultralytics import YOLO
# from app.models.event import EventLog


# import base64
# import cv2
# import numpy as np
# from flask_socketio import SocketIO, emit



# cap = None
# processing_thread = None


# live_stream_bp = Blueprint('live_stream', __name__)
# logger = logging.getLogger(__name__)

# # Global variables for tracking and models (need to be loaded once)
# model = YOLO("yolov8n.pt") # Load model once
# tracker = Sort() # Initialize tracker once
# kalman_tracker = VehicleTracker() # Initialize Kalman tracker once
# try:
#     anomaly_model = joblib.load("app/frontier_anomaly_model.pkl")
# except FileNotFoundError:
#     logging.error("Anomaly detection model not found at app/frontier_anomaly_model.pkl")
#     anomaly_model = None # Handle missing model
# except Exception as e:
#     logging.error(f"Error loading anomaly model: {e}")
#     anomaly_model = None
    
# try:
#     scaler = joblib.load("app/models/scaler.pkl")
# except FileNotFoundError:
#     logging.error("Scaler not found at app/models/scaler.pkl")
#     scaler = None # Handle missing scaler
# except Exception as e:
#     logging.error(f"Error loading scaler: {e}")
#     scaler = None

# try:
#     with open("app/models/frontier_classifier.pkl", "rb") as f:
#         frontier_clf = pickle.load(f)
# except FileNotFoundError:
#     logging.error("Frontier classification model not found at app/models/frontier_classifier.pkl")
#     frontier_clf = None # Handle missing model
# except Exception as e:
#     logging.error(f"Error loading frontier classification model: {e}")
#     frontier_clf = None

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Assuming socketio is accessible globally or passed
# # socketio = SocketIO(message_queue='redis://') # Placeholder, needs proper app context

# def process_live_stream(camera_id):
#     """Process live video stream"""
#     cap = cv2.VideoCapture(camera_id)
#     if not cap.isOpened():
#         logger.error(f"Failed to open camera: {camera_id}")
#         return

#     FPS = 30
#     FRAME_TIME = 1 / FPS
#     prev_frame = None
#     frame_count = 0
#     prev_tracks = {}
#     ego_gps_history = {} # Assuming this needs to be per stream or managed centrally

#     logging.info(f"Camera opened: FPS={FPS}, FRAME_TIME={FRAME_TIME}")

#     # Use app_context if needed for database operations within this thread
#     from run import app # Import the app instance from run.py
#     with app.app_context():
#         try:
#             while cap.isOpened():
#                 success, frame = cap.read()
#                 if not success:
#                     logger.error("Failed to read frame from camera")
#                     break

#                 frame_count += 1
#                 if frame_count % 2 == 0:  # Process every other frame to reduce load
#                     continue

#                 frame = cv2.resize(frame, (640, 480))
#                 height, width, _ = frame.shape

#                 # Lane detection
#                 lane_lines = detect_lanes(frame)
#                 left_lane_x, right_lane_x, _, _ = get_ego_lane_bounds(lane_lines, width, height)
#                 draw_lanes(frame, lane_lines)

#                 # Get GPS data
#                 current_gps = get_gps_data()
#                 ego_speed = current_gps.get("speed", 0.0) # Use .get with default
#                 motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
#                 prev_frame = frame.copy()

#                 # Calculate ego vehicle speed using GPS data
#                 lat, lon = current_gps.get("latitude", 0.0), current_gps.get("longitude", 0.0)
#                 ego_speed_gps = calculate_speed_from_gps(ego_gps_history, lat, lon, frame_count, FRAME_TIME)

#                 # Draw speedometer
#                 draw_speedometer(frame, ego_speed_gps)

#                 # Object detection and tracking
#                 results = model(frame, verbose=False)[0]
#                 detections = []
#                 for r in results.boxes.data.tolist():
#                     x1, y1, x2, y2, score, class_id = r
#                     if score > 0.5:  # Confidence threshold
#                          # Check if the detected object is one of the target classes (e.g., car, truck, bus)
#                         if int(class_id) in [2, 3, 5, 7]: # COCO classes for vehicles
#                             detections.append([x1, y1, x2, y2, score])


#                 tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 5)))
#                 frontier_vehicle = None
#                 min_distance = float('inf')

#                 for track in tracked_objects:
#                     if len(track) < 5:
#                         continue
#                     # Ensure track has at least 5 elements before unpacking
#                     if len(track) >= 5: # Use >= 5 for safety, although it should be 5 or 6
#                         x1, y1, x2, y2, track_id = map(int, track[:5]) # Take only the first 5 elements
#                         # Optionally, if class_id is needed:
#                         # x1, y1, x2, y2, track_id, class_id = map(int, track[:6]) # If tracking includes class_id

#                         color = (255, 0, 0)
#                         event_type = "Tracked"
#                         ttc = None
#                         vehicle_motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status

#                         # Calculate distance to ego vehicle (simplified - distance from bottom of frame)
#                         distance = height - y2 # This is pixel distance, not real-world distance

#                         # Find the closest vehicle (potential frontier vehicle)
#                         # This assumes vehicles closer to the bottom of the frame are closer to the ego vehicle
#                         # Ensure track_id is not -1 before considering it a frontier vehicle
#                         if distance < min_distance and int(track_id) != -1:
#                              min_distance = distance
#                              frontier_vehicle = track

#                         # Check for critical events related to the frontier vehicle
#                         is_critical_event = False
#                         # Use the frontier_vehicle variable for checks instead of recalculating equality
#                         if frontier_vehicle is not None and np.array_equal(track[:5], frontier_vehicle[:5]) and frontier_clf is not None and anomaly_model is not None and scaler is not None:
#                             color = (0, 255, 0) # Green for frontier vehicle
#                             event_type = "Frontier"
#                             y_center = (y1 + y2) // 2

#                             # Estimate frontier vehicle speed (using pixel displacement)
#                             # Pass necessary arguments to estimate_frontier_speed
#                             frontier_speed_px = estimate_frontier_speed(track_id, y_center, frame_count, FRAME_TIME)
#                             # Convert pixel speed to km/h (requires a proper perspective transformation or estimated ground plane)
#                             # For simplicity, let's use a placeholder conversion or assume pixel speed is proportional to real speed
#                             # A more accurate method would involve using the object's estimated distance and size.
#                             # Placeholder conversion:
#                             PIXELS_PER_METER_EST = 0.1 # This value needs to be calibrated
#                             frontier_speed_kmh = (frontier_speed_px * PIXELS_PER_METER_EST) * 3.6


#                             # Calculate TTC (requires real-world distance and speeds)
#                             # Using pixel distance and estimated speeds will be inaccurate.
#                             # If GPS data is available and accurate for both vehicles, use that.
#                             # Assuming for now we use estimated speeds and pixel distance (for demonstration)
#                             # You would need a function to convert pixel distance to real-world distance.
#                             distance_meters = distance * PIXELS_PER_METER_EST # Placeholder conversion

#                             # Ensure ego_speed is available and not None
#                             if ego_speed is not None: # Check if ego_speed was successfully obtained
#                                 ttc = calculate_ttc(ego_speed, frontier_speed_kmh, distance_meters) # Needs real-world values
#                             else:
#                                 ttc = float('inf') # Cannot calculate without ego speed

#                             if ttc is not None and ttc != float('inf') and ttc < 2.0: # TTC threshold
#                                 event_type = "Near Collision"
#                                 is_critical_event = True

#                             # Predict next position using Kalman filter
#                             x_center = (x1 + x2) // 2
#                             # Pass current position to predict_next_position
#                             pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                             cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1) # Draw predicted position

#                             # Anomaly detection for the frontier vehicle
#                             if scaler is not None and anomaly_model is not None:
#                                 try:
#                                     # Create features DataFrame - adjust columns as needed by your model
#                                     features = pd.DataFrame([[frontier_speed_kmh, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"]) # Dummy values for Acc and Lane_ID
#                                     scaled_features_array = scaler.transform(features) # Scaler expects 2D array
#                                     scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"]) # Convert back to DataFrame if needed by model
#                                     # Check prediction result carefully
#                                     if anomaly_model.predict(scaled_features)[0] == -1: # Anomaly detected based on model output
#                                         event_type = f"{event_type} - Anomaly"
#                                         is_critical_event = True
#                                 except Exception as e:
#                                     logging.error(f"Error during anomaly detection: {e}")


#                             # Motion check (simplified)
#                             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Calculate current center
#                             if track_id in prev_tracks:
#                                 prev_cx, prev_cy = prev_tracks[track_id]
#                                 # Calculate pixel displacement
#                                 displacement_px = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
#                                 # Simple motion status based on displacement threshold (needs calibration)
#                                 if displacement_px < 5 and frontier_speed_kmh < 10: # Small displacement and low speed
#                                     vehicle_motion = "Sudden Stop Detected!"
#                                     is_critical_event = True
#                                 elif displacement_px > 10 and frontier_speed_kmh > 50: # Large displacement and high speed
#                                      vehicle_motion = "Harsh Braking" # This logic might need to be reversed or refined
#                                      is_critical_event = True
#                             prev_tracks[track_id] = (cx, cy) # Update previous track position


#                             # Collision detection (simplified - based on IOU with other tracked objects)
#                             # This is a basic overlap check, not a predictive collision detection.
#                             current_bbox = [x1, y1, x2, y2]
#                             for other_track in tracked_objects:
#                                 # Ensure other_track has enough elements before accessing
#                                 if len(other_track) >= 5 and int(other_track[4]) != track_id:
#                                     other_bbox = [int(other_track[0]), int(other_track[1]), int(other_track[2]), int(other_track[3])]
#                                     if calculate_iou(current_bbox, other_bbox) > 0.2: # IOU threshold for potential collision
#                                         vehicle_motion = "Collided"
#                                         is_critical_event = True
#                                         break # No need to check other objects for this vehicle

#                         # Draw bounding box and labels
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#                         # Add labels for Motion Status, Speed, TTC, and ID
#                         font = cv2.FONT_HERSHEY_SIMPLEX
#                         font_scale = 0.5
#                         font_thickness = 1
#                         label_y_offset = y1 - 10 # Start drawing labels above the bounding box
#                         label_spacing = 20 # Vertical space between labels

#                         # Display Vehicle ID
#                         cv2.putText(frame, f"ID: {track_id}", (x1, label_y_offset), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
#                         label_y_offset += label_spacing

#                         # Display Motion Status
#                         cv2.putText(frame, f"Motion: {vehicle_motion}", (x1, label_y_offset), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # Corrected f-string
#                         label_y_offset += label_spacing

#                         # Display Speed (only for frontier vehicle, or estimate for others if needed)
#                         # Check if frontier_vehicle is the current track AND frontier_speed_kmh was calculated
#                         if frontier_vehicle is not None and np.array_equal(track[:5], frontier_vehicle[:5]) and 'frontier_speed_kmh' in locals():
#                              cv2.putText(frame, f"Speed: {frontier_speed_kmh:.1f} km/h", (x1, label_y_offset), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # Corrected f-string
#                              label_y_offset += label_spacing

#                         # Display TTC (only for frontier vehicle)
#                         # Check if frontier_vehicle is the current track AND ttc was calculated
#                         if frontier_vehicle is not None and np.array_equal(track[:5], frontier_vehicle[:5]) and ttc is not None:
#                              ttc_text = f"TTC: {ttc:.1f}s" if ttc != float('inf') else "TTC: N/A"
#                              cv2.putText(frame, ttc_text, (x1, label_y_offset), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # Corrected f-string

#                         # Log and emit critical events
#                         if is_critical_event:
#                             # Create EventLog entry - ensure all required fields are populated
#                             # Import db here if not globally accessible in this thread's context
#                             from app.extensions import db # Import db within the app_context
#                             event = EventLog(
#                                 # Make sure all required fields for EventLog are provided
#                                 # vehicle_id is not a column in the current EventLog model, remove or add it to the model.
#                                 # event_type=event_type, # e.g., "Near Collision", "Frontier - Anomaly"
#                                 # description=f"Critical event detected for vehicle {track_id}: {event_type}", # Add a default description or get from analysis
#                                 # severity="High" if is_critical_event else "Low", # Set severity based on criticality
#                                 # location=f"Lat: {current_gps.get('latitude')}, Lon: {current_gps.get('longitude')}", # Use location from GPS
#                                 # status="new", # Set initial status
                                
#                                 # Based on the current EventLog model:
#                                 event_type=event_type,
#                                 description=f"Critical event detected for vehicle {track_id}: {event_type}. Motion Status: {vehicle_motion}.", # More descriptive
#                                 severity="High", # Assuming critical events are High severity
#                                 location=f"Lat: {current_gps.get('latitude')}, Lon: {current_gps.get('longitude')}" if current_gps.get('latitude') is not None else "Unknown Location", # Provide location if available
#                                 # status defaults to 'pending' in the model definition
#                             )
#                             db.session.add(event) # Add event to the database session

#                             # Emit event data via SocketIO
#                             event_data = {
#                                 # vehicle_id is not in the current EventLog model's to_dict
#                                 # "vehicle_id": track_id,
#                                 "event_type": event_type,
#                                 "timestamp": datetime.utcnow().isoformat(), # Use ISO format for consistency
#                                 # Bounding box coordinates are not in the current EventLog model's to_dict
#                                 # "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                                 "ttc": "N/A" if ttc is None or ttc == float("inf") else round(ttc, 2), # Provide TTC if calculated
#                                 "latitude": current_gps.get("latitude"),
#                                 "longitude": current_gps.get("longitude"),
#                                 # motion_status is not in the current EventLog model's to_dict
#                                 # "motion_status": vehicle_motion,
#                                 "is_critical": is_critical_event # Indicate if it's a critical event
#                             }
#                             try:
#                                 # Use socketio.emit with namespace if applicable, and broadcast=True if needed
#                                 socketio.emit('new_event', event_data) # Emit event to connected clients
#                                 logging.debug(f"Emitted new event: {event_data}")
#                             except Exception as e:
#                                 logging.error(f"Error emitting new event via socketio: {e}")


#                 # Commit database changes periodically
#                 if frame_count % 30 == 0: # Commit every 30 frames
#                     try:
#                         db.session.commit() # Commit the session
#                         logging.debug("Database session committed.")
#                     except Exception as e:
#                         db.session.rollback() # Rollback in case of error
#                         logging.error(f"Database commit failed: {e}")

#                 # Encode and send frame via SocketIO
#                 success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#                 if success:
#                     frame_bytes = buffer.tobytes()
#                     try:
#                         # Emit the frame data via SocketIO
#                         # You might need to encode the frame_bytes to base64 if sending over standard SocketIO
#                         # For simplicity, emitting bytes directly, frontend might need adjustment
#                         socketio.emit('frame', frame_bytes) # Emit frame as bytes
#                         # logging.debug("Frame emitted.") # Avoid excessive logging
#                     except Exception as e:
#                          logging.error(f"Error emitting frame via socketio: {e}")



#         except Exception as e:
#             logging.error(f"Error in process_live_stream: {e}")
#         finally:
#             cap.release()
#             # Clean up database session
#             db.session.remove() # Use db.session.remove() to clean up the session
#             logging.info("Live stream processing stopped.")

# @socketio.on('client_frame')
# def handle_client_frame(base64_img):
#     try:
#         header, encoded = base64_img.split(',', 1)
#         img_bytes = base64.b64decode(encoded)
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if frame is None:
#             print("Failed to decode frame")
#             return

#         # Process frame
#         processed_frame = your_process_function(frame)

#         # Encode to JPEG
#         ret, jpeg = cv2.imencode('.jpg', processed_frame)
#         if not ret:
#             print("Failed to encode frame")
#             return

#         b64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
#         b64_data = 'data:image/jpeg;base64,' + b64_frame

#         emit('processed_frame', b64_data)
#     except Exception as e:
#         print(f"Error processing frame: {e}")


# @socketio.on('start_live_processing')
# def handle_start_live_processing(data):
#     logger.info(f"Received start_live_processing signal with data: {data}")
#     cameraIndex = data.get('cameraIndex')
#     if cameraIndex is None:
#         logger.error("No cameraIndex provided in start_live_processing signal")
#         # Emit an error back to the frontend if no index is provided
#         socketio.emit('stream_error', {'error': 'No camera index provided.'})
#         return

#     # Ensure cameraIndex is an integer
#     try:
#         cameraIndex = int(cameraIndex)
#     except ValueError:
#         logger.error(f"Invalid cameraIndex received: {cameraIndex}")
#         # Emit an error back to the frontend for invalid index
#         socketio.emit('stream_error', {'error': f'Invalid camera index received: {cameraIndex}.'})
#         return

#     global cap
#     # Attempt to open the camera using the integer index
#     logger.info(f"Attempting to open camera with index: {cameraIndex}")
#     cap = cv2.VideoCapture(cameraIndex) # Use the integer index here

#     if not cap.isOpened():
#         logger.error(f"Failed to open camera with index: {cameraIndex}")
#         # Emit an error back to the frontend
#         socketio.emit('stream_error', {'error': f'Failed to open camera with index: {cameraIndex}.'})
#         return

#     logger.info(f"Successfully opened camera with index: {cameraIndex}")
#     # Start a new thread for processing and streaming frames
#     global processing_thread
#     if processing_thread is None or not processing_thread.is_alive():
#         # Start live stream processing in a background thread
#         # Pass the camera_id to the processing function (though process_live_stream also uses cap directly)
#         # We should pass the cameraIndex or cap itself if needed in process_live_stream
#         # Let's pass the cameraIndex for consistency, though process_live_stream currently opens cap again
#         # To avoid opening twice, we should pass the opened cap to process_live_stream

#         # Let's adjust process_live_stream to accept the opened cap
#         logger.info("Starting live stream processing thread.")
#         # Make sure process_live_stream can accept the cap object or the index to open it internally
#         # Based on its current definition process_live_stream(camera_id), it expects an ID/index to open cap internally.
#         # We will keep passing the cameraIndex and modify process_live_stream to use the passed index.

#         threading.Thread(target=process_live_stream, args=(cameraIndex,), daemon=True).start() # Pass the index
#         socketio.emit('status', {'message': f'Started processing for camera index {cameraIndex}'})
#         logging.info(f"Received start_live_processing request for camera index {cameraIndex}")


# @socketio.on('stop_live_processing')
# def handle_stop_live_processing():
#     """Handle stop of live processing"""
#     # In a real application, you would need a way to signal the processing thread to stop gracefully.
#     # This might involve a shared flag or event.
#     # For this example, we'll just log that a stop request was received.
#     logging.info("Received stop_live_processing request. Processing will stop when the current video or stream ends.")
#     # You might want to implement a mechanism to stop the video processing thread here.
#     # For instance, by setting a flag that the process_live_stream function checks.
#     socketio.emit('status', {'message': 'Stop signal received. Processing will end shortly.'})

# @live_stream_bp.route('/live')
# def live():
#     return render_template('live_dashboard.html')

# @live_stream_bp.route('/api/stream/status', methods=['GET'])
# def get_stream_status():
#     try:
#         # In a real application, this would check the actual stream status
#         return jsonify({
#             'status': 'active',
#             'timestamp': datetime.utcnow().isoformat(),
#             'viewers': 0,
#             'quality': 'high'
#         })
#     except Exception as e:
#         logger.error(f"Error getting stream status: {str(e)}")
#         return jsonify({'error': 'Failed to get stream status'}), 500

# @live_stream_bp.route('/api/stream/control', methods=['POST'])
# def control_stream():
#     try:
#         data = request.get_json()
#         action = data.get('action')
        
#         if action not in ['start', 'stop', 'pause']:
#             return jsonify({'error': 'Invalid action'}), 400
            
#         # In a real application, this would control the actual stream
#         socketio.emit('stream_control', {'action': action, 'timestamp': datetime.utcnow().isoformat()})
#         return jsonify({'status': 'success', 'action': action})
#     except Exception as e:
#         logger.error(f"Error controlling stream: {str(e)}")
#         return jsonify({'error': 'Failed to control stream'}), 500

# @live_stream_bp.route('/api/stream/settings', methods=['PUT'])
# def update_stream_settings():
#     try:
#         data = request.get_json()
#         settings = {
#             'quality': data.get('quality', 'high'),
#             'fps': data.get('fps', 30),
#             'resolution': data.get('resolution', '1080p')
#         }
        
#         # In a real application, this would update the actual stream settings
#         socketio.emit('stream_settings_updated', settings)
#         return jsonify(settings)
#     except Exception as e:
#         logger.error(f"Error updating stream settings: {str(e)}")
#         return jsonify({'error': 'Failed to update stream settings'}), 500 

from flask import Blueprint, render_template, jsonify
import cv2
import threading
import logging
import numpy as np
import pandas as pd
import torch
import joblib
import math
import base64
from datetime import datetime
from flask_socketio import emit
from app.extensions import db, socketio
from app.core.sort import Sort
from app.core.vehicle_tracker import VehicleTracker
from app.core.motion_detection import detect_motion_changes
from app.core.utils import (
    calculate_ttc, estimate_frontier_speed, detect_lanes, get_ego_lane_bounds,
    draw_lanes, calculate_iou, calculate_speed_from_gps, draw_speedometer
)
from app.core.gps_module import get_gps_data
import pickle

live_stream_bp = Blueprint('live_stream', __name__)
logger = logging.getLogger(__name__)

# Globals for camera capture and threading control
cap = None
processing_thread = None
stop_event = threading.Event()

# Load models once
device = "cuda" if torch.cuda.is_available() else "cpu"
# Adjust model loading as per your actual setup
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
except ImportError:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n').to(device)

tracker = Sort()
kalman_tracker = VehicleTracker()

try:
    anomaly_model = joblib.load("app/frontier_anomaly_model.pkl")
except Exception as e:
    logger.error(f"Anomaly model load error: {e}")
    anomaly_model = None

try:
    scaler = joblib.load("app/models/scaler.pkl")
except Exception as e:
    logger.error(f"Scaler load error: {e}")
    scaler = None

try:
    with open("app/models/frontier_classifier.pkl", "rb") as f:
        frontier_clf = pickle.load(f)
except Exception as e:
    logger.error(f"Frontier classifier load error: {e}")
    frontier_clf = None


def process_frame(frame, frame_count, prev_frame, prev_tracks, ego_gps_history):
    """Process a single frame, run detection, tracking, annotation, return annotated frame and updated states."""

    height, width = frame.shape[:2]

    # Lane detection and drawing
    lane_lines = detect_lanes(frame)
    left_lane_x, right_lane_x, _, _ = get_ego_lane_bounds(lane_lines, width, height)
    draw_lanes(frame, lane_lines)

    gps = get_gps_data()
    ego_speed = gps.get("speed", 0.0)
    motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
    prev_frame = frame.copy()

    ego_speed_gps = calculate_speed_from_gps(
        ego_gps_history,
        gps.get("latitude", 0.0),
        gps.get("longitude", 0.0),
        frame_count,
        1/30,
    )

    draw_speedometer(frame, ego_speed_gps, width - 80, height - 80, radius=50)

    results = model(frame)[0]
    detections = [
        [int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3])]
        for b in results.boxes if int(b.cls[0]) in [2, 3, 5, 7]
    ]

    tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

    test_data = []
    for t in tracked_objects:
        if len(t) >= 5:
            x1, y1, x2, y2, tid = map(int, t[:5])
            cx = (x1 + x2) // 2
            cy = y2
            w = x2 - x1
            h = y2 - y1
            dist = height - y2
            in_lane = 1 if left_lane_x <= cx <= right_lane_x else 0
            rel_x = (cx - left_lane_x) / (right_lane_x - left_lane_x) if right_lane_x > left_lane_x else 0.5
            test_data.append([x1, y1, x2, y2, cx, cy, w, h, dist, in_lane, rel_x])

    predictions = frontier_clf.predict(test_data) if test_data else []
    frontier_idx = np.argmax(predictions) if np.any(predictions) else -1
    frontier_vehicle = tracked_objects[frontier_idx] if 0 <= frontier_idx < len(tracked_objects) else None

    for t in tracked_objects:
        if len(t) < 5:
            continue
        x1, y1, x2, y2, tid = map(int, t[:5])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = (255, 0, 0)
        event_type = "Tracked"
        motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status
        ttc = None
        frontier_speed = 0
        is_critical_event = False

        if frontier_vehicle is not None and np.array_equal(t, frontier_vehicle):
            color = (0, 255, 0)
            event_type = "Frontier"
            frontier_speed = estimate_frontier_speed(tid, cy, frame_count, 1/30)
            distance = height - y2
            ttc = calculate_ttc(ego_speed, frontier_speed, distance)
            if ttc < 2:
                event_type = "Near Collision"

            pred_x, pred_y = kalman_tracker.predict_next_position(cx, cy)
            cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

            features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
            scaled = scaler.transform(features)
            if anomaly_model.predict(scaled)[0] == -1:
                event_type += " - Anomaly"

            if tid in prev_tracks:
                dx = np.linalg.norm(np.subtract((cx, cy), prev_tracks[tid]))
                if dx < 0.5:
                    motion = "Sudden Stop Detected!"
                elif dx > 5.0:
                    motion = "Harsh Braking"
            prev_tracks[tid] = (cx, cy)

            is_collision = any(
                calculate_iou([x1, y1, x2, y2], [int(o[0]), int(o[1]), int(o[2]), int(o[3])]) > 0.5
                for o in tracked_objects if not np.array_equal(o, t) and len(o) >= 5
            )
            if is_collision:
                motion = "Collided"

            is_critical_event = motion in ["Collided", "Sudden Stop Detected!", "Harsh Braking"] or "Anomaly" in event_type

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th, pad, spacing = 0.4, 1, 3, 15
        label_y = y1 - 80

        labels = [
            f"Motion: {motion}",
            f"Speed: {frontier_speed:.1f} km/h",
            f"TTC: {ttc:.1f}s" if ttc and ttc != float('inf') else "TTC: N/A",
            f"ID: {tid}"
        ]

        for label in labels:
            (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
            bg_pos1 = (x1 - pad, label_y - tht - pad)
            bg_pos2 = (x1 + tw + pad, label_y + pad)
            bg_color = (0, 0, 255) if is_critical_event else (0, 0, 0)
            cv2.rectangle(frame, bg_pos1, bg_pos2, bg_color, -1)
            cv2.putText(frame, label, (x1, label_y), font, fs, (255, 255, 255), th)
            label_y += spacing

    return frame, prev_frame, prev_tracks, ego_gps_history


def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
        return

    frame_count = 0
    prev_frame = None
    prev_tracks = {}
    ego_gps_history = {}

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.info("End of video stream reached or failed to read frame")
                break

            frame_count += 1
            if frame_count % 2 == 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            frame, prev_frame, prev_tracks, ego_gps_history = process_frame(frame, frame_count, prev_frame, prev_tracks, ego_gps_history)

            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ret:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
    finally:
        cap.release()


def process_live_stream(camera_id):
    global cap
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera: {camera_id}")
        return

    frame_count = 0
    prev_frame = None
    prev_tracks = {}
    ego_gps_history = {}

    from run import app
    with app.app_context():
        try:
            while cap.isOpened() and not stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    logger.error("Failed to read frame from camera")
                    break

                frame_count += 1
                if frame_count % 2 == 0:  # Process every other frame
                    continue

                frame = cv2.resize(frame, (640, 480))
                frame, prev_frame, prev_tracks, ego_gps_history = process_frame(frame, frame_count, prev_frame, prev_tracks, ego_gps_history)

                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ret:
                    frame_bytes = base64.b64encode(buffer).decode("utf-8")
                    b64_data = f"data:image/jpeg;base64,{frame_bytes}"
                    try:
                        socketio.emit("processed_frame", b64_data)
                    except Exception as e:
                        logger.error(f"Socket emit error: {e}")

                if frame_count % 30 == 0:
                    try:
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"DB commit error: {e}")

        except Exception as e:
            logger.error(f"Error in live stream processing: {e}")
        finally:
            cap.release()
            db.session.remove()
            logger.info("Live stream processing stopped")


@socketio.on("start_live_processing")
def handle_start_live_processing(data):
    global processing_thread, stop_event
    stop_event.clear()

    camera_index = data.get("cameraIndex")
    if camera_index is None:
        socketio.emit("stream_error", {"error": "No camera index provided."})
        return

    try:
        camera_index = int(camera_index)
    except Exception:
        socketio.emit("stream_error", {"error": "Invalid camera index."})
        return

    logger.info(f"Starting live processing on camera {camera_index}")
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_live_stream, args=(camera_index,), daemon=True)
        processing_thread.start()
        socketio.emit("status", {"message": f"Started processing camera {camera_index}"})


@socketio.on("stop_live_processing")
def handle_stop_live_processing():
    global stop_event
    logger.info("Stop live processing requested")
    stop_event.set()
    socketio.emit("status", {"message": "Stop signal received. Stopping live processing."})


@socketio.on("client_frame")
def handle_client_frame(base64_img):
    try:
        header, encoded = base64_img.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode client frame")
            return

        # If you want continuity across frames, store prev_frame etc in global/session as needed
        processed_frame, _, _, _ = process_frame(frame, 0, None, {}, {})

        ret, jpeg = cv2.imencode(".jpg", processed_frame)
        if not ret:
            logger.error("Failed to encode processed frame")
            return

        b64_frame = base64.b64encode(jpeg.tobytes()).decode("utf-8")
        b64_data = f"data:image/jpeg;base64,{b64_frame}"
        emit("processed_frame", b64_data)
    except Exception as e:
        logger.error(f"Error processing client frame: {e}")


@live_stream_bp.route("/live")
def live():
    return render_template("live_dashboard.html")


@live_stream_bp.route("/api/stream/status", methods=["GET"])
def get_stream_status():
    try:
        return jsonify(
            {
                "status": "active",
                "timestamp": datetime.utcnow().isoformat(),
                "viewers": 0,
                "quality": "high",
            }
        )
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        return jsonify({"error": "Failed to get stream status"}), 500


@live_stream_bp.route("/api/stream/control", methods=["POST"])
def control_stream():
    try:
        data = request.get_json()
        action = data.get("action")
        if action not in ["start", "stop", "pause"]:
            return jsonify({"error": "Invalid action"}), 400
        socketio.emit("stream_control", {"action": action, "timestamp": datetime.utcnow().isoformat()})
        return jsonify({"status": "success", "action": action})
    except Exception as e:
        logger.error(f"Error controlling stream: {e}")
        return jsonify({"error": "Failed to control stream"}), 500


@live_stream_bp.route("/api/stream/settings", methods=["PUT"])
def update_stream_settings():
    try:
        data = request.get_json()
        settings = {
            "quality": data.get("quality", "high"),
            "fps": data.get("fps", 30),
            "resolution": data.get("resolution", "1080p"),
        }
        socketio.emit("stream_settings_updated", settings)
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error updating stream settings: {e}")
        return jsonify({"error": "Failed to update stream settings"}), 500
