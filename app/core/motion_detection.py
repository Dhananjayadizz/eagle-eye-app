# import cv2
# import numpy as np
#
# def detect_motion_changes(prev_frame, curr_frame):
#     """Detects motion changes using Optical Flow (Farneback method)."""
#     if prev_frame is None or curr_frame is None:
#         return "⚠️ No Frame Data"
#
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
#
#     # Compute Optical Flow
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
#                                         0.5, 3, 15, 3, 5, 1.2, 0)
#
#     # Compute magnitude of flow vectors
#     motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
#
#     # Define thresholds for anomaly detection
#     if motion_magnitude > 10:
#         return "🚨 Sudden Stop Detected!"
#     elif motion_magnitude > 5:
#         return "⚠️ Harsh Braking"
#     else:
#         return "✅ Normal Motion"


import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_motion_changes(prev_frame, current_frame):
    """
    Detect motion changes between frames and determine motion status.
    
    Args:
        prev_frame: Previous frame (numpy array)
        current_frame: Current frame (numpy array)
        
    Returns:
        str: Motion status ("Normal Motion", "Harsh Braking", or "Sudden Stop Detected!")
    """
    try:
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(prev_gray, current_gray)
        
        # Apply threshold to identify motion pixels
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion ratio
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_ratio = motion_pixels / total_pixels
        
        # Determine motion status based on thresholds
        if motion_ratio < 0.01:  # Very little motion
            return "Sudden Stop Detected!"
        elif motion_ratio > 0.1:  # Significant motion
            return "Harsh Braking"
        else:
            return "Normal Motion"
            
    except Exception as e:
        logger.error(f"Error in motion detection: {str(e)}")
        return "Normal Motion"  # Default to normal motion on error