import cv2
import numpy as np
import math
from datetime import datetime

def calculate_ttc(ego_speed, frontier_speed, distance):
    """Calculate Time To Collision (TTC)"""
    if ego_speed <= frontier_speed:
        return float('inf')
    relative_speed = ego_speed - frontier_speed
    if relative_speed <= 0:
        return float('inf')
    return distance / relative_speed

def estimate_frontier_speed(track_id, y_center, frame_count, frame_time):
    """Estimate speed of frontier vehicle"""
    # Placeholder implementation
    return 40.0  # km/h

def detect_lanes(frame):
    """Detect lane lines in the frame"""
    # Placeholder implementation
    return []

def get_ego_lane_bounds(lane_lines, width, height):
    """Get ego vehicle lane boundaries"""
    # Placeholder implementation
    return 0, width, 0, height

def draw_lanes(frame, lane_lines):
    """Draw detected lane lines"""
    # Placeholder implementation
    return frame

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def calculate_speed_from_gps(gps_history, lat, lon, frame_count, frame_time):
    """Calculate speed from GPS coordinates"""
    if not gps_history:
        gps_history[frame_count] = (lat, lon)
        return 0.0
    
    prev_frame = max(k for k in gps_history.keys() if k < frame_count)
    prev_lat, prev_lon = gps_history[prev_frame]
    
    distance = haversine_distance(prev_lat, prev_lon, lat, lon)
    time_diff = (frame_count - prev_frame) * frame_time
    
    gps_history[frame_count] = (lat, lon)
    return (distance / time_diff) * 3600  # Convert to km/h

def draw_speedometer(frame, speed):
    """Draw speedometer on frame"""
    cv2.putText(frame, f"Speed: {speed:.1f} km/h", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame 