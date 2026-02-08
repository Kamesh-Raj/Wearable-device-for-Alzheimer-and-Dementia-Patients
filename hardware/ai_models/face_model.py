"""
Face Recognition with Personal Fine-Tuning
Handles face detection, embedding generation, training, and recognition.
Optimized for Raspberry Pi Zero 2W using TFLite or dlib encodings.
"""

import os
import cv2
import numpy as np
import pickle
import logging
from datetime import datetime
# Note: On Pi, install face_recognition via piwheels or use dlib
import face_recognition

class FaceRecognizer:
    def __init__(self, model_dir="hardware/models/face_recognition"):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.encodings_file = os.path.join(model_dir, "known_faces.pkl")
        self.known_encodings = []
        self.known_names = []
        self.known_types = []  # 'patient', 'caregiver', 'known_person'
        
        # Create model directory if not exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing model
        self.load_model()

    def load_model(self):
        """Load trained encodings from disk"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_names = data['names']
                    self.known_types = data.get('types', [])
                self.logger.info(f"Loaded {len(self.known_names)} known faces.")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
        else:
            self.logger.info("No trained model found. Please run fine-tuning.")

    def fine_tune(self, data_dir="hardware/training_data"):
        """
        Personal Fine-tuning:
        Process all images in training directories and rebuild the model.
        Structure:
            data_dir/
                patient/
                caregiver/
                known/
        """
        self.logger.info("Starting personal fine-tuning...")
        temp_encodings = []
        temp_names = []
        temp_types = []

        categories = {
            'patient': os.path.join(data_dir, 'patient'),
            'caregiver': os.path.join(data_dir, 'caregiver'),
            'known': os.path.join(data_dir, 'known')
        }

        for user_type, path in categories.items():
            if not os.path.exists(path):
                continue
                
            for filename in os.listdir(path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(path, filename)
                    name = os.path.splitext(filename)[0].split('_')[0] # Assume name_id.jpg
                    
                    try:
                        image = face_recognition.load_image_file(image_path)
                        # Detect faces
                        boxes = face_recognition.face_locations(image, model="hog")
                        
                        if len(boxes) != 1:
                            self.logger.warning(f"Image {filename} has {len(boxes)} faces, skipping.")
                            continue
                            
                        # Generate encoding
                        encoding = face_recognition.face_encodings(image, boxes)[0]
                        
                        temp_encodings.append(encoding)
                        temp_names.append(name)
                        temp_types.append(user_type)
                        self.logger.info(f"Processed {name} ({user_type})")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {filename}: {e}")

        if not temp_encodings:
            self.logger.warning("No valid training data found.")
            return

        # Save model
        data = {
            "encodings": temp_encodings,
            "names": temp_names,
            "types": temp_types,
            "trained_at": datetime.now().isoformat()
        }
        
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
            
        self.known_encodings = temp_encodings
        self.known_names = temp_names
        self.known_types = temp_types
        self.logger.info("Fine-tuning complete. Model saved.")

    def recognize(self, frame):
        """
        Recognize faces in a frame.
        Returns list of dicts: {'name': str, 'type': str, 'location': tuple, 'confidence': float}
        """
        # Resize for speed optimization on Pi
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        results = []

        for face_encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            user_type = "unknown"
            confidence = 0.0

            if self.known_encodings:
                # Calculate distances to all known faces
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                # Threshold for match
                if face_distances[best_match_index] < 0.6:
                    name = self.known_names[best_match_index]
                    user_type = self.known_types[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

            # Scale location back up
            top, right, bottom, left = location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            results.append({
                "name": name,
                "type": user_type,
                "location": (top, right, bottom, left),
                "confidence": confidence
            })

        return results