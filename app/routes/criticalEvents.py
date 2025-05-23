from flask import Blueprint, render_template, jsonify, request, send_file, current_app
from app.extensions import db, socketio
from app.data.event_log import EventLog
from datetime import datetime
import logging
import pandas as pd
import io
import os
import cv2
from ultralytics import YOLO
import numpy as np
from app.core.utils import calculate_ttc, estimate_frontier_speed, detect_lanes, get_ego_lane_bounds, draw_lanes, calculate_iou, haversine_distance, calculate_speed_from_gps, draw_speedometer
from app.core.gps_module import get_gps_data
from app.core.motion_detection import detect_motion_changes
import re


# Flask + Sockets
from flask import Flask, request, render_template, jsonify, Response, send_file
from flask_socketio import SocketIO, emit

# Computer Vision & Math
import cv2
import numpy as np
import math
import re

# Deep Learning
from ultralytics import YOLO
import torch

# Model I/O
import joblib
import pickle

# Data Handling
import pandas as pd
import io
import openpyxl
from datetime import datetime

# Database
from flask_sqlalchemy import SQLAlchemy

# File Handling
import os
import shutil
from pathlib import Path
import uuid

# Threads & Serial
import threading
import time
import serial
from threading import Lock

# Logging
import logging


# App Core
from app.core.sort import Sort
from app.core.vehicle_tracker import VehicleTracker
from app.core.gps_module import get_gps_data
from app.core.motion_detection import detect_motion_changes

from threading import Lock

# Global GPS data and lock
gps_data = {
    "latitude": 0.0,
    "longitude": 0.0,
    "connected": False
}
gps_lock = Lock()


critical_events_bp = Blueprint('critical_events', __name__)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

PIXELS_PER_METER = 0.1
vehicle_history = {}
collided_vehicles = set()
collision_cooldown = {}
ego_gps_history = {}  # To store GPS history for the ego vehicle
frontier_gps_history = {}  # To store GPS history for the frontier vehicle


serial_port = None
SERIAL_PORT_NAME = 'COM5' # Set the serial port name to COM5
BAUDRATE = 115200 # Match the Serial.begin() baud rate in your Arduino sketch

# Replace the MAX_STORED_VIDEOS constant with a single video file name
CURRENT_VIDEO_FILE = "current_video.mp4"

# Initialize YOLO model
model = YOLO('yolov8n.pt')




@critical_events_bp.route('/section1')
def critical_events():
    return render_template('critical_events_analysis.html')


def read_gps_data_from_serial(port, baudrate):
    global serial_port
    logger.info(f"Connecting to GPS on {port} at {baudrate}...")

    try:
        serial_port = serial.Serial(port, baudrate, timeout=1)
        logger.info("Serial port opened.")
        with gps_lock:
            gps_data["connected"] = True

        while True:
            if serial_port.in_waiting > 0:
                line = serial_port.readline().decode('utf-8', errors='ignore').strip()
                logger.debug(f"Raw GPS line: {line}")

                lat_match = re.search(r'Latitude:(-?\d+\.\d+)', line)
                lon_match = re.search(r'Longitude:(-?\d+\.\d+)', line)

                if lat_match:
                    with gps_lock:
                        gps_data['latitude'] = float(lat_match.group(1))
                        gps_data['connected'] = True

                if lon_match:
                    with gps_lock:
                        gps_data['longitude'] = float(lon_match.group(1))
                        gps_data['connected'] = True

                    # Emit once both are likely to be updated
                    socketio.emit('gps_update', gps_data)

            time.sleep(0.1)

    except Exception as e:
        logger.error(f"GPS Serial Error: {e}")
        with gps_lock:
            gps_data.update({
                "latitude": 0.0,
                "longitude": 0.0,
                "connected": False
            })
        socketio.emit('gps_update', gps_data)



# Load models
model = YOLO("yolov8n.pt").to(device)
tracker = Sort()
kalman_tracker = VehicleTracker()
anomaly_model = joblib.load("app/models/frontier_anomaly_model.pkl")
scaler = joblib.load("app/models/scaler.pkl")
try:
    with open("app/models/frontier_classifier.pkl", "rb") as f:
        frontier_clf = pickle.load(f)
    logger.info("Frontier vehicle classification model loaded successfully.")
except FileNotFoundError:
    logger.error("Frontier classification model 'frontier_classifier.pkl' not found.")
    raise
except Exception as e:
    logger.error(f"Error loading frontier classification model: {e}")
    raise

# Haversine formula to calculate distance between two GPS coordinates (in meters)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance


# Calculate speed from GPS coordinates over time using frame count
def calculate_speed_from_gps(gps_history, lat, lon, frame_count, frame_time):
    key = "ego"  # Use a fixed key for the ego vehicle
    if key not in gps_history:
        gps_history[key] = {"last_lat": lat, "last_lon": lon, "last_frame": frame_count, "speed": 40.0}
        return 40.0

    last_lat = gps_history[key]["last_lat"]
    last_lon = gps_history[key]["last_lon"]
    last_frame = gps_history[key]["last_frame"]
    time_diff = (frame_count - last_frame) * frame_time

    if time_diff <= 0:
        logger.info(f"Time difference zero or negative for ego vehicle")
        return gps_history[key]["speed"]

    distance = haversine_distance(last_lat, last_lon, lat, lon)
    speed_mps = distance / time_diff
    speed_kmh = speed_mps * 3.6
    speed_kmh = max(0, min(120, speed_kmh))

    alpha = 0.7
    smoothed_speed = alpha * speed_kmh + (1 - alpha) * gps_history[key]["speed"]
    gps_history[key]["speed"] = smoothed_speed
    gps_history[key]["last_lat"] = lat
    gps_history[key]["last_lon"] = lon
    gps_history[key]["last_frame"] = frame_count
    logger.info(f"Calculated speed for ego vehicle: {smoothed_speed} km/h")
    return smoothed_speed

# Draw a cyan speedometer with a transparent background as a 270-degree arc with numbers on the arc
def draw_speedometer(frame, speed, center_x=None, center_y=None, radius=60):
    CYAN = (95, 189, 255)  # Define cyan color explicitly

    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Set the speedometer position to the bottom-right corner
    margin = 20  # Margin from the edges
    center_x = width - radius - margin  # Position center_x near the right edge
    center_y = height - radius - margin  # Position center_y near the bottom edge

    # Draw the outer arc of the speedometer (270 degrees, from 315° to 225° counterclockwise)
    start_angle = 315    # Start at 7:30 position (315°)
    end_angle = 225      # End at 4:30 position (225°), covering 270° counterclockwise
    cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, start_angle, end_angle, CYAN, 2)

    # Draw speed markers and numbers on the arc
    for speed_mark in range(0, 121, 20):
        # Map speed (0-120) to angle (315° to 225° counterclockwise), starting from 0 at 315° to 120 at 225°
        angle = math.radians(315 - (speed_mark / 120.0) * 270)  # From 315° to 225° (270° range)
        x1 = int(center_x + (radius - 5) * math.cos(angle))  # Inner point of the marker
        y1 = int(center_y - (radius - 5) * math.sin(angle))
        x2 = int(center_x + radius * math.cos(angle))  # Outer point of the marker (on the arc)
        y2 = int(center_y - radius * math.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), CYAN, 1)

        # Place the number exactly on the arc
        label_x = int(center_x + radius * math.cos(angle))  # Position exactly on the arc
        label_y = int(center_y - radius * math.sin(angle))

        # Adjust text position based on angle to center the numbers
        text = str(speed_mark)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        # Calculate offset to center the text on the arc
        offset_x = -text_width // 2  # Center the text horizontally
        offset_y = text_height // 2  # Center the text vertically
        adjusted_x = label_x + offset_x
        adjusted_y = label_y + offset_y

        # Add a subtle black outline to the text for better visibility
        cv2.putText(frame, text, (adjusted_x, adjusted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, text, (adjusted_x, adjusted_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)  # Cyan text

    # Draw the needle in cyan
    speed = min(max(speed, 0), 120)  # Clamp speed between 0 and 120
    angle = math.radians(315 - (speed / 120.0) * 270)  # Map speed from 315° (0 km/h) to 225° (120 km/h)
    needle_length = radius - 10
    needle_x = int(center_x + needle_length * math.cos(angle))
    needle_y = int(center_y - needle_length * math.sin(angle))
    cv2.line(frame, (center_x, center_y), (needle_x, needle_y), CYAN, 2)

    # Draw the speed text in the top-right corner of the video feed
    speed_text = f"{int(speed)} km/h"
    (text_width, text_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_pos = (width - text_width - 20, 30)  # Position in top-right corner (20 pixels from right, 30 from top)
    cv2.putText(frame, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
    cv2.putText(frame, speed_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)  # Cyan text



def calculate_ttc(ego_speed, frontier_speed, distance):
    if frontier_speed <= ego_speed or distance <= 0:
        return float('inf')
    relative_speed = (ego_speed - frontier_speed) / 3.6
    return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')

def estimate_frontier_speed(track_id, y_center, frame_count, frame_time):
    if track_id not in vehicle_history:
        vehicle_history[track_id] = {"last_y": y_center, "last_frame": frame_count, "speed": 40.0}
        return 40.0
    last_y = vehicle_history[track_id]["last_y"]
    last_frame = vehicle_history[track_id]["last_frame"]
    time_diff = (frame_count - last_frame) * frame_time
    if time_diff > 0:
        displacement = last_y - y_center
        speed_pixels_per_sec = displacement / time_diff
        speed_mps = speed_pixels_per_sec * PIXELS_PER_METER
        speed_kmh = speed_mps * 3.6
        alpha = 0.7
        new_speed = max(0, min(120, speed_kmh))
        smoothed_speed = alpha * new_speed + (1 - alpha) * vehicle_history[track_id]["speed"]
        vehicle_history[track_id]["speed"] = smoothed_speed
    vehicle_history[track_id]["last_y"] = y_center
    vehicle_history[track_id]["last_frame"] = frame_count
    return vehicle_history[track_id]["speed"]


def detect_lanes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]

    # Color masks for white and yellow lanes
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    lower_yellow = np.array([15, 50, 100])
    upper_yellow = np.array([35, 255, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 50, 150)

    # Apply region of interest (trapezoid mask)
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[  # Trapezoid mask
        [width * 0.1, height * 0.9],
        [width * 0.4, height * 0.55],
        [width * 0.6, height * 0.55],
        [width * 0.9, height * 0.9]
    ]], np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    masked_edges = cv2.bitwise_and(masked_edges, color_mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=40)

    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3:  # filter out near-horizontal lines
                    lane_lines.append((x1, y1, x2, y2))
    return lane_lines



def get_ego_lane_bounds(lane_lines, width, height):
    if not lane_lines:
        return 0, width, None, None

    left_lane_x = width
    right_lane_x = 0
    left_lines = []
    right_lines = []
    left_line_points = []
    right_line_points = []

    for x1, y1, x2, y2 in lane_lines:
        if y1 != y2:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)

            if slope > 0.5:
                right_lines.append(x_bottom)
                right_line_points.append((x1, y1, x2, y2))
            elif slope < -0.5:
                left_lines.append(x_bottom)
                left_line_points.append((x1, y1, x2, y2))

    if left_lines:
        left_lane_x = max(0, min(left_lines) - 50)
    if right_lines:
        right_lane_x = min(width, max(right_lines) + 50)

    return (int(left_lane_x),
            int(right_lane_x),
            left_line_points if left_line_points else None,
            right_line_points if right_line_points else None)

def draw_lanes(frame, lane_lines):
    if lane_lines:
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lanes


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_other, y1_other, x2_other, y2_other = box2
    xi1 = max(x1, x1_other)
    yi1 = max(y1, y1_other)
    xi2 = min(x2, x2_other)
    yi2 = min(y2, y2_other)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_other - x1_other) * (y2_other - y1_other)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


@critical_events_bp.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Create uploads directory if it doesn't exist
        # upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        basedir = os.path.abspath(os.path.dirname(__file__))
        upload_dir = os.path.join(basedir, '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Save the video file
        video_path = os.path.join(upload_dir, 'current_video.mp4')
        video_file.save(video_path)
        
        # Get the app instance from the current request context
        app = current_app._get_current_object()
        
        # Start video processing in a background thread with app context
        socketio.start_background_task(process_video_with_context, app, video_path)
        
        return jsonify({
            'success': True,
            'video_path': video_path
        })
    except Exception as e:
        logger.error(f"Error uploading video: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_video_with_context(app, video_path):
    """Wrapper function to provide application context for video processing"""
    process_video(video_path)

# Modify the video_feed route to use the current video
# @critical_events_bp.route("/video_feed")
# def video_feed():
#     video_path = os.path.join(app.config["UPLOAD_FOLDER"], CURRENT_VIDEO_FILE)
#     if not os.path.exists(video_path):
#         return jsonify({'error': 'No video available'}), 404
        
#     logger.info(f"Starting video feed for: {video_path}")
#     return Response(process_video(video_path),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

from flask import stream_with_context

@critical_events_bp.route("/video_feed")
def video_feed():
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
    video_path = os.path.join(upload_dir, CURRENT_VIDEO_FILE)
    
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return jsonify({'error': 'No video available'}), 404

    logger.info(f"Starting video feed for: {video_path}")
    return Response(stream_with_context(process_video(video_path)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# def process_video(video_path):
#     cap = None
#     try:
#         cap = cv2.VideoCapture(video_path)
#         frame_count = 0
#         prev_frame = None
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             # Resize frame for processing
#             frame = cv2.resize(frame, (640, 480))
            
#             # Get GPS data
#             gps = get_gps_data()
            
#             # Detect motion changes
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
            
#             # Perform object detection
#             results = model(frame)
            
#             # Process detections
#             for result in results:
#                 boxes = result.boxes
#                 for box in boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = float(box.conf[0])
#                     cls = int(box.cls[0])
                    
#                     if conf > 0.5:  # Confidence threshold
#                         # Calculate TTC
#                         ttc = calculate_ttc(ego_speed=40.0,  # Placeholder ego speed
#                                          frontier_speed=estimate_frontier_speed(track_id=cls,
#                                                                               y_center=(y1 + y2) / 2,
#                                                                               frame_count=frame_count,
#                                                                               frame_time=frame_count/30),
#                                          distance=calculate_distance(x1, y1, x2, y2))
                        
#                         # Create event log
#                         event = EventLog(
#                             vehicle_id=cls,
#                             event_type="Vehicle Detected",
#                             x1=x1, y1=y1, x2=x2, y2=y2,
#                             ttc=ttc,
#                             latitude=gps["latitude"],
#                             longitude=gps["longitude"],
#                             motion_status=motion_status
#                         )
                        
#                         db.session.add(event)
                        
#                         # Emit event through socket
#                         socketio.emit('new_event', event.to_dict())
            
#             # Commit to database every 30 frames
#             if frame_count % 30 == 0:
#                 try:
#                     db.session.commit()
#                 except Exception as e:
#                     logger.error(f"Commit failed: {e}")
            
#             prev_frame = frame.copy()
            
#     except Exception as e:
#         logger.error(f"Error processing video: {str(e)}")
#     finally:
#         if cap is not None:
#             cap.release()
#         try:
#             if os.path.exists(video_path):
#                 os.remove(video_path)
#         except Exception as e:
#             logger.error(f"Error removing video file: {str(e)}")

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logger.error(f"Failed to open video: {video_path}")
#         yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
#         return

#     FPS = cap.get(cv2.CAP_PROP_FPS)
#     FRAME_TIME = 1 / FPS
#     frame_count = 0
#     prev_frame = None
#     prev_tracks = {}

#     try:
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 logger.info("End of video stream reached")
#                 break

#             frame_count += 1
#             if frame_count % 2 == 0:
#                 continue

#             frame = cv2.resize(frame, (640, 480))
#             height, width = frame.shape[:2]

#             # Draw ego lane
#             lane_lines = detect_lanes(frame)
#             left_lane_x, right_lane_x, _, _ = get_ego_lane_bounds(lane_lines, width, height)

#             for x1, y1, x2, y2 in lane_lines:
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Draw ego lane bounds
#             cv2.line(frame, (left_lane_x, height), (left_lane_x, int(height * 0.6)), (0, 0, 255), 2)
#             cv2.line(frame, (right_lane_x, height), (right_lane_x, int(height * 0.6)), (0, 0, 255), 2)

#             gps = get_gps_data()
#             ego_speed = gps["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
#             prev_frame = frame.copy()

#             ego_speed_gps = calculate_speed_from_gps(ego_gps_history, gps["latitude"], gps["longitude"], frame_count, FRAME_TIME)
#             draw_speedometer(frame, ego_speed_gps, width - 80, height - 80, radius=50)

#             results = model(frame)[0]
#             detections = [[int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3])]
#                           for b in results.boxes if int(b.cls[0]) in [2, 3, 5, 7]]
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

#             test_data = []
#             for t in tracked_objects:
#                 if len(t) >= 5:
#                     x1, y1, x2, y2, tid = map(int, t)
#                     cx = (x1 + x2) // 2
#                     cy = y2
#                     w = x2 - x1
#                     h = y2 - y1
#                     dist = height - y2
#                     in_lane = 1 if left_lane_x <= cx <= right_lane_x else 0
#                     rel_x = (cx - left_lane_x) / (right_lane_x - left_lane_x) if right_lane_x > left_lane_x else 0.5
#                     test_data.append([x1, y1, x2, y2, cx, cy, w, h, dist, in_lane, rel_x])

#             predictions = frontier_clf.predict(test_data) if test_data else []
#             frontier_idx = np.argmax(predictions) if np.any(predictions) else -1
#             frontier_vehicle = tracked_objects[frontier_idx] if 0 <= frontier_idx < len(tracked_objects) else None

#             for t in tracked_objects:
#                 if len(t) < 5:
#                     continue
#                 x1, y1, x2, y2, tid = map(int, t)
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status
#                 ttc, frontier_speed = None, 0
#                 is_critical_event = False

#                 if np.array_equal(t, frontier_vehicle):
#                     color = (0, 255, 0)
#                     event_type = "Frontier"
#                     frontier_speed = estimate_frontier_speed(tid, cy, frame_count, FRAME_TIME)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     pred_x, pred_y = kalman_tracker.predict_next_position(cx, cy)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

#                     # Anomaly detection
#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled = scaler.transform(features)
#                     if anomaly_model.predict(scaled)[0] == -1:
#                         event_type += " - Anomaly"

#                     # Motion check
#                     if tid in prev_tracks:
#                         dx = np.linalg.norm(np.subtract((cx, cy), prev_tracks[tid]))
#                         if dx < 0.5:
#                             motion = "Sudden Stop Detected!"
#                         elif dx > 5.0:
#                             motion = "Harsh Braking"
#                     prev_tracks[tid] = (cx, cy)

#                     # Collision detection
#                     is_collision = any(
#                         calculate_iou([x1, y1, x2, y2], [int(o[0]), int(o[1]), int(o[2]), int(o[3])]) > 0.5
#                         for o in tracked_objects if not np.array_equal(o, t) and len(o) >= 5
#                     )
#                     if is_collision:
#                         motion = "Collided"

#                     is_critical_event = motion in ["Collided", "Sudden Stop Detected!", "Harsh Braking"] or "Anomaly" in event_type or "Collision" in event_type

#                 # Labels
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 fs = 0.4
#                 th = 1
#                 pad = 3
#                 spacing = 15
#                 motion_text = f"Motion: {motion}"
#                 speed_text = f"Speed: {frontier_speed:.1f} km/h" if frontier_speed else "Speed: N/A"
#                 ttc_text = f"TTC: {ttc:.1f}s" if ttc and ttc != float('inf') else "TTC: N/A"
#                 id_text = f"ID: {tid}"


#                 labels = [motion_text, speed_text, ttc_text, id_text]
#                 label_positions = [(x1, y1 - 80 + spacing * i) for i in range(len(labels))]

#                 overlay = frame.copy()
#                 for i, (text, pos) in enumerate(zip(labels, label_positions)):
#                     (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
#                     bg_pos1 = (pos[0] - pad, pos[1] - tht - pad)
#                     bg_pos2 = (pos[0] + tw + pad, pos[1] + pad)
#                     bg_color = (0, 0, 255) if is_critical_event else (0, 0, 0)
#                     cv2.rectangle(overlay, bg_pos1, bg_pos2, bg_color, -1)
#                     cv2.putText(overlay, text, pos, font, fs, (255, 255, 255), th)

#                 cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#                 # Log and emit
#                 event = EventLog(vehicle_id=tid, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=None if ttc == float("inf") else ttc,
#                                  latitude=gps["latitude"], longitude=gps["longitude"],
#                                  motion_status=motion)
#                 if is_critical_event:
#                     logger.info(f"Critical event detected: {event_type} for vehicle ID {tid} with motion status {motion}")
#                     db.session.add(event)

#                     # Only emit critical events
#                 try:
#                     event_data = {
#                         "id": event.id,
#                         "vehicle_id": tid,
#                         "event_type": event_type,
#                         "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
#                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                         "ttc": "N/A" if not ttc or ttc == float("inf") else round(ttc, 2),
#                         "latitude": gps["latitude"], "longitude": gps["longitude"],
#                         "motion_status": motion,
#                         "is_critical": is_critical_event
#                     }
#                     logger.debug(f"Emitting event for vehicle ID {tid} (Critical: {is_critical_event})")
#                     socketio.emit("new_event", event_data)
#                     logger.info(f"Emitted event data: {event_data}")
#                 except Exception as e:
#                     logger.error(f"Socket emit error: {e}")

#             if frame_count % 30 == 0:
#                 try:
#                     db.session.commit()
#                 except Exception as e:
#                     logger.error(f"Commit failed: {e}")

#             success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#             if success:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     except Exception as e:
#         logger.error(f"process_video crashed: {e}")
#     finally:
#         cap.release()


# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logger.error(f"Failed to open video: {video_path}")
#         yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
#         return

#     FPS = cap.get(cv2.CAP_PROP_FPS)
#     FRAME_TIME = 1 / FPS
#     frame_count = 0
#     prev_frame = None
#     prev_tracks = {}
#     event_counter = 0 # Initialize event counter for SocketIO emissions

#     try:
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 logger.info("End of video stream reached")
#                 break

#             frame_count += 1
#             if frame_count % 2 == 0:
#                 continue

#             frame = cv2.resize(frame, (640, 480))
#             height, width = frame.shape[:2]

#             lane_lines = detect_lanes(frame)
#             left_lane_x, right_lane_x, _, _ = get_ego_lane_bounds(lane_lines, width, height)

#             for x1, y1, x2, y2 in lane_lines:
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # cv2.line(frame, (left_lane_x, height), (left_lane_x, int(height * 0.6)), (0, 0, 255), 2)
#             # cv2.line(frame, (right_lane_x, height), (right_lane_x, int(height * 0.6)), (0, 0, 255), 2)

#             gps = get_gps_data()
#             ego_speed = gps["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
#             prev_frame = frame.copy()

#             ego_speed_gps = calculate_speed_from_gps(ego_gps_history, gps["latitude"], gps["longitude"], frame_count, FRAME_TIME)
#             draw_speedometer(frame, ego_speed_gps, width - 80, height - 80, radius=50)

#             results = model(frame)[0]
#             detections = [[int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3])]
#                           for b in results.boxes if int(b.cls[0]) in [2, 3, 5, 7]]
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

#             test_data = []
#             for t in tracked_objects:
#                 if len(t) >= 5:
#                     x1, y1, x2, y2, tid = map(int, t)
#                     cx = (x1 + x2) // 2
#                     cy = y2
#                     w = x2 - x1
#                     h = y2 - y1
#                     dist = height - y2
#                     in_lane = 1 if left_lane_x <= cx <= right_lane_x else 0
#                     rel_x = (cx - left_lane_x) / (right_lane_x - left_lane_x) if right_lane_x > left_lane_x else 0.5
#                     test_data.append([x1, y1, x2, y2, cx, cy, w, h, dist, in_lane, rel_x])

#             predictions = frontier_clf.predict(test_data) if test_data else []
#             frontier_idx = np.argmax(predictions) if np.any(predictions) else -1
#             frontier_vehicle = tracked_objects[frontier_idx] if 0 <= frontier_idx < len(tracked_objects) else None

#             for t in tracked_objects:
#                 if len(t) < 5:
#                     continue
#                 x1, y1, x2, y2, tid = map(int, t)
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status
#                 ttc, frontier_speed = None, 0
#                 is_critical_event = False

#                 if np.array_equal(t, frontier_vehicle):
#                     color = (0, 255, 0)
#                     event_type = "Frontier"
#                     frontier_speed = estimate_frontier_speed(tid, cy, frame_count, FRAME_TIME)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     pred_x, pred_y = kalman_tracker.predict_next_position(cx, cy)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)

#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled = scaler.transform(features)
#                     if anomaly_model.predict(scaled)[0] == -1:
#                         event_type += " - Anomaly"

#                     if tid in prev_tracks:
#                         dx = np.linalg.norm(np.subtract((cx, cy), prev_tracks[tid]))
#                         if dx < 0.5:
#                             motion = "Sudden Stop Detected!"
#                         elif dx > 5.0:
#                             motion = "Harsh Braking"
#                     prev_tracks[tid] = (cx, cy)

#                     is_collision = any(
#                         calculate_iou([x1, y1, x2, y2], [int(o[0]), int(o[1]), int(o[2]), int(o[3])]) > 0.5
#                         for o in tracked_objects if not np.array_equal(o, t) and len(o) >= 5
#                     )
#                     if is_collision:
#                         motion = "Collided"

#                     is_critical_event = motion in ["Collided", "Sudden Stop Detected!", "Harsh Braking"] or "Anomaly" in event_type

#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 fs = 0.4
#                 th = 1
#                 pad = 3
#                 spacing = 15
#                 motion_text = f"Motion: {motion}"
#                 speed_text = f"Speed: {frontier_speed:.1f} km/h" if frontier_speed else "Speed: N/A"
#                 ttc_text = f"TTC: {ttc:.1f}s" if ttc and ttc != float('inf') else "TTC: N/A"
#                 id_text = f"ID: {tid}"

#                 labels = [motion_text, speed_text, ttc_text, id_text]
#                 label_positions = [(x1, y1 - 80 + spacing * i) for i in range(len(labels))]

#                 overlay = frame.copy()
#                 for i, (text, pos) in enumerate(zip(labels, label_positions)):
#                     (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
#                     bg_pos1 = (pos[0] - pad, pos[1] - tht - pad)
#                     bg_pos2 = (pos[0] + tw + pad, pos[1] + pad)
#                     bg_color = (0, 0, 255) if is_critical_event else (0, 0, 0)
#                     cv2.rectangle(overlay, bg_pos1, bg_pos2, bg_color, -1)
#                     cv2.putText(overlay, text, pos, font, fs, (255, 255, 255), th)

#                 cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#                 event = EventLog(vehicle_id=tid, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=None if ttc == float("inf") else ttc,
#                                  latitude=gps["latitude"], longitude=gps["longitude"],
#                                  motion_status=motion)

#                 if is_critical_event:
#                     db.session.add(event)
#                     db.session.flush() # Assigns an ID without committing the transaction

#                 try:
#                     event_counter += 1 # Increment counter for each emitted event
#                     event_data = {
#                         "id": event_counter, # Use the counter-based ID for SocketIO
#                         "vehicle_id": tid,
#                         "event_type": event_type,
#                         "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
#                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                         "ttc": None if not ttc or ttc == float("inf") else round(ttc, 2),
#                         "latitude": gps["latitude"], "longitude": gps["longitude"],
#                         "motion_status": motion,
#                         "is_critical": is_critical_event
#                     }
#                     logger.debug(f"Emitting event for vehicle ID {tid} (Critical: {is_critical_event})")
#                     socketio.emit("new_event", event_data)
#                     logger.info(f"Emitted event data: {event_data}")
#                 except Exception as e:
#                     logger.error(f"Socket emit error: {e}")

#             if frame_count % 30 == 0:
#                 try:
#                     db.session.commit()
#                 except Exception as e:
#                     logger.error(f"Commit failed: {e}")

#             success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#             if success:
#                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     except Exception as e:
#         logger.error(f"process_video crashed: {e}")
#     finally:
#         cap.release()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
        return

    FPS = cap.get(cv2.CAP_PROP_FPS)
    FRAME_TIME = 1 / FPS
    frame_count = 0
    prev_frame = None
    prev_tracks = {}
    event_counter = 0  # For SocketIO event IDs

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.info("End of video stream reached")
                break

            frame_count += 1
            if frame_count % 2 == 0:
                continue  # Skip every other frame to reduce load

            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]

            # --- Lane detection and drawing with fallback ---
            lane_lines = detect_lanes(frame)
            left_lane_x, right_lane_x, left_points, right_points = get_ego_lane_bounds(lane_lines, width, height)

            if not lane_lines or (left_points is None and right_points is None):
                # Fallback: draw red polygon if no lanes detected
                roi_vertices = np.array([[  
                    [int(width * 0.1), int(height * 0.9)],
                    [int(width * 0.4), int(height * 0.55)],
                    [int(width * 0.6), int(height * 0.55)],
                    [int(width * 0.9), int(height * 0.9)]
                ]], np.int32)
                cv2.polylines(frame, roi_vertices, isClosed=True, color=(0, 0, 255), thickness=1)
                logger.warning("No lane lines detected; drawing fallback ROI polygon")
            else:
                # Draw detected lane lines (green)
                for x1, y1, x2, y2 in lane_lines:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Draw left lane boundary lines (blue)
                if left_points:
                    for x1, y1, x2, y2 in left_points:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        logger.debug(f"Drawing left lane boundary line: ({x1},{y1}) to ({x2},{y2})")

                # Draw right lane boundary lines (red)
                if right_points:
                    for x1, y1, x2, y2 in right_points:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        logger.debug(f"Drawing right lane boundary line: ({x1},{y1}) to ({x2},{y2})")

            gps = get_gps_data()
            ego_speed = gps["speed"]
            motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "Normal Motion"
            prev_frame = frame.copy()

            ego_speed_gps = calculate_speed_from_gps(ego_gps_history, gps["latitude"], gps["longitude"], frame_count, FRAME_TIME)
            draw_speedometer(frame, ego_speed_gps, width - 80, height - 80, radius=50)

            results = model(frame)[0]
            detections = [[int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3])]
                          for b in results.boxes if int(b.cls[0]) in [2, 3, 5, 7]]
            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

            test_data = []
            for t in tracked_objects:
                if len(t) >= 5:
                    x1, y1, x2, y2, tid = map(int, t)
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
                x1, y1, x2, y2, tid = map(int, t)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = (255, 0, 0)
                event_type = "Tracked"
                motion = "Normal Motion" if motion_status == "Normal Motion" else motion_status
                ttc, frontier_speed = None, 0
                is_critical_event = False

                if np.array_equal(t, frontier_vehicle):
                    color = (0, 255, 0)
                    event_type = "Frontier"
                    frontier_speed = estimate_frontier_speed(tid, cy, frame_count, FRAME_TIME)
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

                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = 0.4
                th = 1
                pad = 3
                spacing = 15
                motion_text = f"Motion: {motion}"
                speed_text = f"Speed: {frontier_speed:.1f} km/h" if frontier_speed else "Speed: N/A"
                ttc_text = f"TTC: {ttc:.1f}s" if ttc and ttc != float('inf') else "TTC: N/A"
                id_text = f"ID: {tid}"

                labels = [motion_text, speed_text, ttc_text, id_text]
                label_positions = [(x1, y1 - 80 + spacing * i) for i in range(len(labels))]

                overlay = frame.copy()
                for i, (text, pos) in enumerate(zip(labels, label_positions)):
                    (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
                    bg_pos1 = (pos[0] - pad, pos[1] - tht - pad)
                    bg_pos2 = (pos[0] + tw + pad, pos[1] + pad)
                    bg_color = (0, 0, 255) if is_critical_event else (0, 0, 0)
                    cv2.rectangle(overlay, bg_pos1, bg_pos2, bg_color, -1)
                    cv2.putText(overlay, text, pos, font, fs, (255, 255, 255), th)

                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                event = EventLog(vehicle_id=tid, event_type=event_type,
                                 x1=x1, y1=y1, x2=x2, y2=y2, ttc=None if ttc == float("inf") else ttc,
                                 latitude=gps["latitude"], longitude=gps["longitude"],
                                 motion_status=motion)

                if is_critical_event:
                    db.session.add(event)
                    db.session.flush()  # Assigns an ID without committing the transaction

                try:
                    event_counter += 1  # Increment counter for each emitted event
                    event_data = {
                        "id": event_counter,  # Use the counter-based ID for SocketIO
                        "vehicle_id": tid,
                        "event_type": event_type,
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "ttc": None if not ttc or ttc == float("inf") else round(ttc, 2),
                        "latitude": gps["latitude"], "longitude": gps["longitude"],
                        "motion_status": motion,
                        "is_critical": is_critical_event
                    }
                    logger.debug(f"Emitting event for vehicle ID {tid} (Critical: {is_critical_event})")
                    socketio.emit("new_event", event_data)
                    logger.info(f"Emitted event data: {event_data}")
                except Exception as e:
                    logger.error(f"Socket emit error: {e}")

            if frame_count % 30 == 0:
                try:
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Commit failed: {e}")

            success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if success:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        logger.error(f"process_video crashed: {e}")
    finally:
        cap.release()



@critical_events_bp.route('/export_critical_events', methods=['GET'])
def export_critical_events():
    try:
        critical_event_types = ["Collided", "Harsh Braking", "Sudden Stop Detected!"]
        critical_events = EventLog.query.filter(EventLog.motion_status.in_(critical_event_types)).all()

        if not critical_events:
            return jsonify({"error": "No critical events found."}), 404

        data = []
        for event in critical_events:
            data.append({
                "ID": event.id,
                "Vehicle ID": event.vehicle_id,
                "Event Type": event.event_type,
                "Motion Status": event.motion_status,
                "Timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "X1": event.x1, "Y1": event.y1, "X2": event.x2, "Y2": event.y2,
                "TTC (s)": "N/A" if event.ttc is None or event.ttc == float('inf') else event.ttc,
                "Latitude": event.latitude,
                "Longitude": event.longitude
            })

        # Export to Excel
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Critical Events")

        filename = f"critical_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def calculate_distance(x1, y1, x2, y2):
    """Calculate distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) 


@critical_events_bp.route("/events", methods=["GET"])
def get_events():
    events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
    logger.info(f"Events retrieved: {len(events)}")
    data = [{
        "id": e.id,
        "vehicle_id": e.vehicle_id,
        "event_type": e.event_type,
        "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if e.timestamp else "N/A",
        "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
        "ttc": "N/A" if e.ttc is None or e.ttc == float('inf') else e.ttc,
        "latitude": e.latitude,
        "longitude": e.longitude,
        "motion_status": e.motion_status
    } for e in events]
    logger.debug(f"Data returned: {data}")
    return jsonify(data)

# @critical_events_bp.route("/export_critical_events", methods=["GET"])
# def export_critical_events():
#     try:
#         with app.app_context():
#             critical_event_types = ["Collided", "Harsh Braking", "Sudden Stop Detected!"]
#             critical_events = EventLog.query.filter(EventLog.motion_status.in_(critical_event_types)).all()

#             if not critical_events:
#                 logger.info("No critical events found.")
#                 return jsonify({"error": "No critical events found."}), 404

#             data = []
#             for event in critical_events:
#                 data.append({
#                     "ID": event.id,
#                     "Vehicle ID": event.vehicle_id,
#                     "Event Type": event.event_type,
#                     "Motion Status": event.motion_status,
#                     "Timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.timestamp else "N/A",
#                     "X1": event.x1, "Y1": event.y1, "X2": event.x2, "Y2": event.y2,
#                     "TTC (s)": "N/A" if event.ttc is None or event.ttc == float('inf') else event.ttc,
#                     "Latitude": event.latitude,
#                     "Longitude": event.longitude
#                 })

#             df = pd.DataFrame(data)
#             output = io.BytesIO()
#             with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                 df.to_excel(writer, index=False, sheet_name="Critical Events")

#             # Define the new directory for blockchain excels
#             BLOCKCHAIN_EXCELS_DIR = os.path.join(os.path.dirname(app.instance_path), "blockchain_excles")
#             os.makedirs(BLOCKCHAIN_EXCELS_DIR, exist_ok=True)

#             # Save the file to the new directory
#             filename = f"critical_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
#             filepath = os.path.join(BLOCKCHAIN_EXCELS_DIR, filename)
#             with open(filepath, "wb") as f:
#                 f.write(output.getvalue())

#             logger.info(f"Critical events exported to {filepath}")
#             return jsonify({"message": "Critical events exported successfully to blockchain_excles folder."})

#     except Exception as e:
#         logger.error(f"Error exporting critical events: {e}")
#         return jsonify({"error": "Failed to export critical events."}), 500










# @critical_events_bp.route("/list_exported_files", methods=["GET"])
# def list_exported_files():
#     try:
#         files = []
#         for idx, filename in enumerate(os.listdir(EXPORT_DIR)):
#             if filename.endswith('.xlsx'):
#                 file_path = os.path.join(EXPORT_DIR, filename)
#                 timestamp = os.path.getmtime(file_path)
#                 files.append({
#                     "id": idx + 1,
#                     "file_name": filename,
#                     "timestamp": int(timestamp * 1000)
#                 })
#         logger.info(f"Listed {len(files)} exported files")
#         return jsonify({"files": files})
#     except Exception as e:
#         logger.error(f"Error listing exported files: {e}")
#         return jsonify({"error": "Failed to list exported files"}), 500



def get_gps_data():
    with gps_lock:
        return {
            "latitude": gps_data["latitude"],
            "longitude": gps_data["longitude"],
            "speed": 40.0 if gps_data["connected"] else 0.0,  # Placeholder
            "connected": gps_data["connected"]
        }


@critical_events_bp.route("/delete_exported_file", methods=["POST"])
def delete_exported_file():
    try:
        data = request.get_json()
        filename = data.get("filename")
        if not filename:
            logger.error("No filename provided")
            return jsonify({"error": "No filename provided"}), 400

        file_path = os.path.join(EXPORT_DIR, filename)
        if os.path.exists(file_path) and filename.endswith('.xlsx'):
            os.remove(file_path)
            logger.info(f"File deleted: {filename}")
            return jsonify({"message": "File deleted successfully"})
        else:
            logger.error(f"File not found or invalid: {filename}")
            return jsonify({"error": "File not found or invalid"}), 404
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"error": "Failed to delete file"}), 500

@critical_events_bp.route("/clear_exported_files", methods=["POST"])
def clear_exported_files():
    try:
        for filename in os.listdir(EXPORT_DIR):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(EXPORT_DIR, filename)
                os.remove(file_path)
        logger.info("All exported files cleared")
        return jsonify({"message": "All exported files cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing exported files: {e}")
        return jsonify({"error": "Failed to clear exported files"}), 500


# @critical_events_bp.route('/video_feed')
# def video_feed():
#     video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads', 'latest_video.mp4')
#     if not os.path.exists(video_path):
#         return jsonify({'error': 'Video not found'}), 404
    
#     return send_file(video_path, mimetype='video/mp4')


@critical_events_bp.route('/uploads/<filename>')
def serve_uploaded_video(filename):
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
    video_path = os.path.join(upload_dir, filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(video_path, mimetype='video/mp4')


