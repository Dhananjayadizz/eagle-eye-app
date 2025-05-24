import io
import logging
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from ultralytics import YOLO

from app.extensions import db, socketio

# Optional S3 support - will be handled gracefully if boto3 is not installed
try:
    import tempfile
    from io import BytesIO

    import boto3
    S3_AVAILABLE = True
    try:
        # Test if Textract is available
        textract_client = boto3.client('textract')
        TEXTRACT_AVAILABLE = True
    except Exception as e:
        logging.warning(f"AWS Textract not available: {str(e)}")
        TEXTRACT_AVAILABLE = False
except ImportError:
    S3_AVAILABLE = False
    TEXTRACT_AVAILABLE = False
    logging.warning("boto3 not installed. S3 upload and Textract functionality will be disabled.")

# Create Blueprint
traffic_analysis_bp = Blueprint('traffic_analysis', __name__)
logger = logging.getLogger(__name__)

# Database Models
class BaseEvent(db.Model):
    """Base model for all detection events"""
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    frame_number = db.Column(db.Integer, nullable=False)
    x1 = db.Column(db.Integer, nullable=False)
    y1 = db.Column(db.Integer, nullable=False)
    x2 = db.Column(db.Integer, nullable=False)
    y2 = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    
    def to_dict(self):
        """Base dictionary conversion"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'frame_number': self.frame_number,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'confidence': self.confidence
        }

class PotholeEvent(BaseEvent):
    """Model for pothole detection events"""
    __tablename__ = 'pothole_events'
    
    pothole_id = db.Column(db.Integer, nullable=True)
    severity = db.Column(db.String(20), nullable=False)
    
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'event_type': 'Pothole',
            'vehicle_id': self.pothole_id,
            'motion_status': self.severity,
            'ttc': None,
            'latitude': 0.0,
            'longitude': 0.0
        })
        return data

class NumberPlateEvent(BaseEvent):
    """Model for number plate detection events"""
    __tablename__ = 'numberplate_events'
    
    vehicle_id = db.Column(db.Integer, nullable=True)
    plate_text = db.Column(db.String(20), nullable=True)
    s3_image_url = db.Column(db.String(255), nullable=True)
    
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'event_type': 'Number Plate',
            'vehicle_id': self.vehicle_id,
            'motion_status': 'Detected' if self.plate_text else 'Unknown',
            'ttc': None,
            'latitude': 0.0,
            'longitude': 0.0
        })
        return data

class StopLightViolationEvent(BaseEvent):
    """Model for stop light violation events"""
    __tablename__ = 'stoplight_events'
    
    vehicle_id = db.Column(db.Integer, nullable=True)
    light_state = db.Column(db.String(10), nullable=False)  # Red, Green
    is_moving = db.Column(db.Boolean, default=False)
    is_violation = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        data = super().to_dict()
        data.update({
            'event_type': 'Stop Light Violation' if self.is_violation else 'Stop Light',
            'vehicle_id': self.vehicle_id,
            'motion_status': 'Moving' if self.is_moving else 'Stopped',
            'ttc': None,
            'latitude': 0.0,
            'longitude': 0.0
        })
        return data

# Constants and Global Variables
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
TRAFFIC_LIGHT_CLASS = "traffic light"

# Load models
pothole_model = None
vehicle_model = None
np_model = None

# Global variables
frame_skip = 2
processing_thread = None
stop_processing = False

# Routes
@traffic_analysis_bp.route('/section2')
def traffic_analysis():
    """Render the traffic analysis page"""
    return render_template('traffic_analysis.html')

@traffic_analysis_bp.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload for traffic detection"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Create uploads directory if it doesn't exist
        basedir = os.path.abspath(os.path.dirname(__file__))
        upload_dir = os.path.join(basedir, '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the video file
        video_path = os.path.join(upload_dir, 'current_traffic_video.mp4')
        video_file.save(video_path)
        
        # Get the app instance from the current request context
        app = current_app._get_current_object()
        
        # Start video processing in a background thread with app context
        global processing_thread, stop_processing
        stop_processing = True  # Stop any existing processing
        if processing_thread and processing_thread.is_alive():
            processing_thread.join(timeout=1.0)
        
        stop_processing = False
        processing_thread = threading.Thread(target=process_video_with_context, args=(app, video_path))
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            'success': True,
            'video_path': video_path
        })
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@traffic_analysis_bp.route('/video_feed')
def video_feed():
    """Video streaming route for traffic detection"""
    basedir = os.path.abspath(os.path.dirname(__file__))
    upload_dir = os.path.join(basedir, '..', 'uploads')
    video_path = os.path.join(upload_dir, 'current_traffic_video.mp4')
    
    if not os.path.exists(video_path):
        return "No video has been uploaded yet.", 404
    
    return Response(process_video(video_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@traffic_analysis_bp.route('/export_events')
def export_events():
    """Export all detection events to Excel"""
    try:
        pothole_events = PotholeEvent.query.all()
        numberplate_events = NumberPlateEvent.query.all()
        stoplight_events = StopLightViolationEvent.query.all()
        
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Pothole events sheet
            if pothole_events:
                pothole_data = [event.to_dict() for event in pothole_events]
                pd.DataFrame(pothole_data).to_excel(writer, index=False, sheet_name='Pothole Events')
            
            # Number plate events sheet
            if numberplate_events:
                np_data = [event.to_dict() for event in numberplate_events]
                pd.DataFrame(np_data).to_excel(writer, index=False, sheet_name='Number Plate Events')
            
            # Stop light violation events sheet
            if stoplight_events:
                sl_data = [event.to_dict() for event in stoplight_events]
                pd.DataFrame(sl_data).to_excel(writer, index=False, sheet_name='Stop Light Events')
        
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name='traffic_detection_events.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Error exporting events: {str(e)}")
        return jsonify({'error': str(e)}), 500

@traffic_analysis_bp.route('/clear_files', methods=['POST'])
def clear_files():
    """Clear uploaded files and reset database"""
    try:
        # Clear database
        PotholeEvent.query.delete()
        NumberPlateEvent.query.delete()
        StopLightViolationEvent.query.delete()
        db.session.commit()
        
        # Clear uploaded files
        basedir = os.path.abspath(os.path.dirname(__file__))
        upload_dir = os.path.join(basedir, '..', 'uploads')
        video_path = os.path.join(upload_dir, 'current_traffic_video.mp4')
        
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Clear any saved number plate images
        np_image_path = os.path.join(upload_dir, 'highest_conf_number_plate.png')
        if os.path.exists(np_image_path):
            os.remove(np_image_path)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing files: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Video Processing Functions
def process_video_with_context(app, video_path):
    """Process video with application context"""
    with app.app_context():
        try:
            # Load models if not already loaded
            global pothole_model, vehicle_model, np_model
            if pothole_model is None:
                pothole_model = YOLO('app/models/best.pt')
                logger.info("Pothole detection model loaded successfully.")
            if vehicle_model is None:
                vehicle_model = YOLO('app/models/yolov8n.pt')
                logger.info("Vehicle detection model loaded successfully.")
            if np_model is None:
                np_model = YOLO('app/models/np_best.pt')
                logger.info("Number plate detection model loaded successfully.")
                
            for frame in process_video(video_path):
                # This just iterates through the generator to process the video
                pass
        except Exception as e:
            logger.error(f"Error in video processing thread: {str(e)}")

def process_video(video_path):
    """Process video for all detection types"""
    global stop_processing
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
        return

    FPS = cap.get(cv2.CAP_PROP_FPS)
    FRAME_TIME = 1 / FPS
    frame_count = 0
    
    # Variables for stop light detection
    vehicle_areas = {}
    global_traffic_light_state = "Unknown"
    prev_gray = None
    
    # Variables for number plate detection
    max_conf = 0
    np_max_frame = None
    
    try:
        while cap.isOpened() and not stop_processing:
            success, frame = cap.read()
            if not success:
                logger.info("End of video stream reached")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip processing some frames
            
            # Resize the frame
            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Check if dashcam is stopped (for stop light detection)
            dashcam_stopped = is_dashcam_stopped(frame, prev_gray)
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Run vehicle detection
            vehicle_results = vehicle_model.track(frame, persist=True)
            vehicle_boxes = vehicle_results[0].boxes
            
            # Process traffic lights and vehicles
            for box in vehicle_boxes:
                CLASS_ID = int(box.cls.item())
                CLASS_NAME = COCO_CLASSES[CLASS_ID] if CLASS_ID < len(COCO_CLASSES) else ""
                CONFIDENCE = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                TRACK_ID = int(box.id.item()) if box.id is not None else None

                if CONFIDENCE < 0.4:
                    continue

                if CLASS_NAME == "traffic light":
                    roi = frame[y1:y2, x1:x2]
                    detected_color = get_traffic_light_color(roi)
                    if detected_color == "Red":
                        global_traffic_light_state = "Red"
                    elif detected_color == "Green":
                        global_traffic_light_state = "Green"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, f"Light: {global_traffic_light_state}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 155), 2)
                    
                    # Create event for database
                    event = StopLightViolationEvent(
                        frame_number=frame_count,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(CONFIDENCE),
                        light_state=global_traffic_light_state,
                        is_moving=False,
                        is_violation=False
                    )
                    
                    # Add to database
                    db.session.add(event)
                    
                    # Emit event through socket
                    event_data = event.to_dict()
                    socketio.emit('new_event', event_data)
                    continue

                # Process vehicles for stop light violations
                if global_traffic_light_state == "Red" and dashcam_stopped:
                    if CLASS_NAME in ["car", "motorcycle", "bus", "truck"] and TRACK_ID is not None:
                        current_area = (x2 - x1) * (y2 - y1)
                        is_moving = is_vehicle_moving(TRACK_ID, current_area, vehicle_areas)
                        
                        box_color = (0, 255, 0)  # Default: Green for not moving
                        if is_moving:
                            box_color = (0, 0, 255)  # Red if moving
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                        label = f"{CLASS_NAME} {TRACK_ID}"
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                        
                        # Create event for database
                        event = StopLightViolationEvent(
                            vehicle_id=TRACK_ID,
                            frame_number=frame_count,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=float(CONFIDENCE),
                            light_state=global_traffic_light_state,
                            is_moving=is_moving,
                            is_violation=is_moving and global_traffic_light_state == "Red"
                        )
                        
                        # Add to database
                        db.session.add(event)
                        
                        # Emit event through socket
                        event_data = event.to_dict()
                        socketio.emit('new_event', event_data)
                
                # Process vehicles for number plate detection
                if CLASS_NAME in ["car", "motorcycle", "bus", "truck"] and TRACK_ID is not None:
                    # Draw vehicle bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{CLASS_NAME} {TRACK_ID}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Crop the vehicle frame and detect number plate
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:  # Skip if crop is empty
                        continue
                        
                    np_results = np_model(vehicle_crop)
                    np_boxes = np_results[0].boxes
                    
                    for np_box in np_boxes:
                        NP_CONFIDENCE = np_box.conf.item()
                        if NP_CONFIDENCE < 0.6:
                            continue
                        
                        np_x1, np_y1, np_x2, np_y2 = map(int, np_box.xyxy[0].tolist())
                        
                        # Adjust coordinates to original frame
                        global_np_x1 = x1 + np_x1
                        global_np_y1 = y1 + np_y1
                        global_np_x2 = x1 + np_x2
                        global_np_y2 = y1 + np_y2
                        
                        cv2.rectangle(display_frame, (global_np_x1, global_np_y1), (global_np_x2, global_np_y2), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Number Plate", (global_np_x1, global_np_y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Save the highest confidence number plate
                        if NP_CONFIDENCE > max_conf:
                            max_conf = NP_CONFIDENCE
                            # Expand and crop from the original frame
                            np_max_frame = expand_and_crop(frame, global_np_x1, global_np_y1, global_np_x2, global_np_y2)
                            
                            # Process the number plate image
                            if np_max_frame is not None and np_max_frame.size > 0:
                                # Save enhanced image
                                plate_text = None
                                s3_url = None
                                
                                # Save locally for OCR
                                basedir = os.path.abspath(os.path.dirname(__file__))
                                upload_dir = os.path.join(basedir, '..', 'uploads')
                                np_image_path = os.path.join(upload_dir, f'plate_{TRACK_ID}_{frame_count}.png')
                                
                                # Enhance and save the image
                                enhanced_np_image = enhance_number_plate_image(np_max_frame)
                                cv2.imwrite(np_image_path, enhanced_np_image)
                                
                                # Try OCR if available
                                if TEXTRACT_AVAILABLE:
                                    try:
                                        plate_text = extract_text_from_image(np_image_path)
                                        logger.info(f"Extracted text from plate: {plate_text}")
                                    except Exception as e:
                                        logger.error(f"Error extracting text: {str(e)}")
                                
                                # Upload to S3 if available
                                if S3_AVAILABLE:
                                    try:
                                        s3_url = upload_image_to_s3(enhanced_np_image, f"plate_{TRACK_ID}_{frame_count}.png")
                                    except Exception as e:
                                        logger.error(f"Error uploading to S3: {str(e)}")
                            
                            # Create event for database
                            event = NumberPlateEvent(
                                vehicle_id=TRACK_ID,
                                frame_number=frame_count,
                                x1=global_np_x1, y1=global_np_y1, x2=global_np_x2, y2=global_np_y2,
                                confidence=float(NP_CONFIDENCE),
                                plate_text=plate_text,
                                s3_image_url=s3_url
                            )
                            
                            # Add to database
                            db.session.add(event)
                            
                            # Emit event through socket
                            event_data = event.to_dict()
                            socketio.emit('new_event', event_data)
            
            # Process pothole detection
            # Define ROI (lower half of the frame)
            roi_start = height // 3
            roi = frame[roi_start:height, :]
            
            # Run object detection on the ROI
            pothole_results = pothole_model.track(roi, persist=True)
            pothole_boxes = pothole_results[0].boxes
            
            # Process each detected pothole
            for box in pothole_boxes:
                CONFIDENCE = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                TRACK_ID = int(box.id.item()) if box.id is not None else -1

                if CONFIDENCE < 0.2:  # Filter out low-confidence detections
                    continue
                
                # Adjust bounding box coordinates relative to the original frame
                y1 += roi_start
                y2 += roi_start

                # Draw red bounding box for potholes
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Pothole: {CONFIDENCE:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Determine severity based on size and confidence
                pothole_size = (x2 - x1) * (y2 - y1)
                size_ratio = pothole_size / (width * height)
                
                severity = "Low"
                if size_ratio > 0.05 and CONFIDENCE > 0.5:
                    severity = "Medium"
                if size_ratio > 0.1 and CONFIDENCE > 0.7:
                    severity = "High"
                
                # Create event for database
                event = PotholeEvent(
                    pothole_id=TRACK_ID,
                    frame_number=frame_count,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(CONFIDENCE),
                    severity=severity
                )
                
                # Add to database
                db.session.add(event)
                
                # Emit event through socket
                event_data = event.to_dict()
                socketio.emit('new_event', event_data)
            
            # Add traffic light state to display
            cv2.putText(display_frame, f"Traffic Light: {global_traffic_light_state}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 155), 2)
            
            if dashcam_stopped:
                cv2.putText(display_frame, "Dashcam Stabilized", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Commit to database every 30 frames
            if frame_count % 30 == 0:
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Database commit failed: {e}")
                    db.session.rollback()
            
            # Convert frame to JPEG for streaming
            success, buffer = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if success:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
    finally:
        if cap.isOpened():
            cap.release()
        try:
            db.session.commit()  # Final commit
        except Exception as e:
            logger.error(f"Final database commit failed: {e}")
            db.session.rollback()

# Helper Functions
def is_dashcam_stopped(frame, prev_gray, threshold=0.4):
    """Checks if the dashcam vehicle is stationary using optical flow."""
    if prev_gray is None:
        return False
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_movement = np.mean(mag)
    
    return avg_movement < threshold

def is_vehicle_moving(track_id, current_area, vehicle_areas, threshold=0.97):
    """Check if a vehicle is moving based on bounding box area decreasing."""
    if track_id in vehicle_areas:
        prev_area = vehicle_areas[track_id]
        area_change = current_area / prev_area if prev_area else 1.0
        vehicle_areas[track_id] = current_area  # Update the stored area
        return area_change < threshold  # If area decreases significantly, it's moving
    else:
        vehicle_areas[track_id] = current_area
        return False

def get_traffic_light_color(roi):
    """Determine the color of a detected traffic light."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([90, 255, 255])
    
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    if np.sum(mask_red) > np.sum(mask_green):
        return "Red"
    elif np.sum(mask_green) > np.sum(mask_red):
        return "Green"
    return "Unknown"

def expand_and_crop(image, x1, y1, x2, y2, scale=2.0):
    """Expands the bbox by a given scale and crops the image."""
    h, w = image.shape[:2]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    box_w = x2 - x1
    box_h = y2 - y1

    new_w = int(box_w * scale)
    new_h = int(box_h * scale)

    new_x1 = max(cx - new_w // 2, 0)
    new_y1 = max(cy - new_h // 2, 0)
    new_x2 = min(cx + new_w // 2, w)
    new_y2 = min(cy + new_h // 2, h)

    return image[new_y1:new_y2, new_x1:new_x2]

def enhance_number_plate_image(image):
    """Enhance number plate image for better OCR."""
    if image is None or image.size == 0:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Resize (2x upscale for better OCR)
    resized = cv2.resize(sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return resized

def extract_text_from_image(image_path):
    """Extract text from image using AWS Textract."""
    if not TEXTRACT_AVAILABLE:
        return None
        
    try:
        # Read the image file
        with open(image_path, 'rb') as document:
            image_bytes = document.read()

        # Create Textract client
        textract = boto3.client('textract')

        # Call Textract
        response = textract.detect_document_text(Document={'Bytes': image_bytes})

        # Extract detected text
        detected_text = []
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                detected_text.append(item['Text'])

        # Join all detected lines
        return ' '.join(detected_text) if detected_text else None
    except Exception as e:
        logger.error(f"Error extracting text with Textract: {str(e)}")
        return None

def upload_image_to_s3(frame, file_name):
    """Upload image to S3 bucket."""
    if not S3_AVAILABLE:
        logger.warning("S3 upload attempted but boto3 is not installed")
        return None
        
    if frame is None or frame.size == 0:
        return None
    
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            logger.error('Failed to encode the frame as an image')
            return None

        image_file = BytesIO(buffer)
        bucket_name = 'traffic-violations-bucket'
        
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(image_file, bucket_name, file_name)
        object_url = f'https://{bucket_name}.s3.amazonaws.com/{file_name}'
        logger.info(f'File uploaded: {object_url}')
        return object_url
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return None
