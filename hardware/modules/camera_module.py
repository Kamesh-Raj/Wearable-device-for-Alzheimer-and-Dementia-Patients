"""
Camera Module with Rotation Handling
Handles camera input with automatic rotation correction for face detection
"""

import cv2
import numpy as np
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None # PC Fallback

from typing import Tuple, Optional
import logging

class CameraModule:
    """
    Camera module for Raspberry Pi with rotation handling.
    Falls back to OpenCV VideoCapture on PC.
    """
    
    def __init__(self, rotation=0, resolution=(640, 480)):
        self.logger = logging.getLogger(__name__)
        self.rotation = rotation
        self.resolution = resolution
        self.camera = None
        self.cap = None # OpenCV fallback
        
        if Picamera2:
            try:
                # Initialize PiCamera2
                self.camera = Picamera2()
                config = self.camera.create_still_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.logger.info(f"PiCamera initialized with rotation: {rotation}°")
            except Exception as e:
                self.logger.error(f"Failed to initialize PiCamera: {e}")
                self.camera = None
        else:
            self.logger.warning("PiCamera2 not found. Using OpenCV VideoCapture (PC Mode).")
            try:
                self.cap = cv2.VideoCapture(0)
                self.logger.info("OpenCV Camera initialized.")
            except Exception as e:
                self.logger.error(f"Failed to init OpenCV camera: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from active camera"""
        if self.camera:
            try:
                return self.camera.capture_array()
            except Exception as e:
                self.logger.error(f"PiCamera capture failed: {e}")
                return None
        elif self.cap:
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        return None
    
    def rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame based on camera orientation
        
        Args:
            frame: Input frame
            
        Returns:
            Rotated frame
        """
        if self.rotation == 0:
            return frame
        elif self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self.logger.warning(f"Invalid rotation {self.rotation}, using 0°")
            return frame
    
    def detect_and_correct_rotation(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Auto-detect face orientation and correct rotation
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (corrected_frame, detected_rotation)
        """
        # Try all rotations and find the one with best face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        best_rotation = 0
        max_faces = 0
        best_frame = frame
        
        for rotation in [0, 90, 180, 270]:
            # Rotate frame
            if rotation == 0:
                test_frame = frame
            elif rotation == 90:
                test_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                test_frame = cv2.rotate(frame, cv2.ROTATE_180)
            else:  # 270
                test_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Detect faces
            gray = cv2.cvtColor(test_frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Track best rotation
            if len(faces) > max_faces:
                max_faces = len(faces)
                best_rotation = rotation
                best_frame = test_frame
        
        return best_frame, best_rotation
    
    def get_frame(self, auto_rotate=False) -> Optional[np.ndarray]:
        """
        Get a frame with rotation applied
        
        Args:
            auto_rotate: If True, auto-detect and correct rotation
            
        Returns:
            Processed frame or None
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        if auto_rotate:
            frame, detected_rotation = self.detect_and_correct_rotation(frame)
            if detected_rotation != self.rotation:
                self.logger.info(f"Auto-corrected rotation from {self.rotation}° to {detected_rotation}°")
        else:
            frame = self.rotate_frame(frame)
        
        return frame
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces.tolist() if len(faces) > 0 else []
    
    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face region from frame
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        # Add padding
        padding = int(0.2 * w)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        return frame[y1:y2, x1:x2]
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            self.camera.stop()
            self.logger.info("Camera released")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize camera with 180° rotation (camera mounted upside down)
    camera = CameraModule(rotation=180)
    
    try:
        # Capture frame with auto-rotation
        frame = camera.get_frame(auto_rotate=True)
        
        if frame is not None:
            # Detect faces
            faces = camera.detect_faces(frame)
            print(f"Detected {len(faces)} face(s)")
            
            # Extract each face
            for i, face_bbox in enumerate(faces):
                face_img = camera.extract_face(frame, face_bbox)
                print(f"Face {i+1} size: {face_img.shape}")
    
    finally:
        camera.release()
