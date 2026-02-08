"""
Gait / Motion Analyzer
Optimized for Raspberry Pi Zero 2W.
Uses lightweight frame differencing to detect activity levels, potential falls, and wandering.
True skeleton-based gait analysis is too heavy for Pi Zero 2W.
"""

import cv2
import numpy as np
import time
import logging

class GaitAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prev_frame = None
        self.min_area = 500 # Minimum contour area
        
        # State
        self.last_motion_time = time.time()
        self.is_moving = False
        self.motion_duration = 0
        
        # Thresholds
        self.fall_threshold = 50 # Vertical motion speed
        self.wandering_time = 300 # 5 minutes of continuous motion

    def analyze(self, frame):
        """
        Analyze motion in frame.
        Returns:
            anomaly: str or None (e.g. 'fall', 'wandering', 'inactive')
            confidence: float
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None, 0.0

        # Frame Delta
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        max_y_change = 0
        
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(c)
            # Simple vertical center check logic could go here
            
        # Update State
        current_time = time.time()
        
        if motion_detected:
            if not self.is_moving:
                self.is_moving = True
                self.motion_start_time = current_time
            
            self.last_motion_time = current_time
            self.motion_duration = current_time - self.motion_start_time
            
            # Check Wandering
            if self.motion_duration > self.wandering_time:
                return "wandering", 0.8
                
        else:
            self.is_moving = False
            self.motion_duration = 0
            
            # Check Inactivity (e.g. 1 hour)
            if (current_time - self.last_motion_time) > 3600:
                 return "inactive", 0.6
                 
        self.prev_frame = gray
        
        # Fall detection requires higher FPS tracking of centroid Y-velocity
        # Placeholder for Pi Zero implementation
        
        return None, 0.0
