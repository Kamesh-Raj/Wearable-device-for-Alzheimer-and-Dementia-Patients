"""
Caregiver Recognition Module
Integrates face recognition with database for known person identification
Uses trained face_recognition_95_int8.tflite model for 95%+ accuracy
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from hardware.ai_models.face_recognition import FaceRecognitionSystem

class CaregiverRecognition:
    """Handles recognition of known caregivers and family members"""
    
    def __init__(self, db_manager, patient_id: int):
        self.logger = logging.getLogger(__name__)
        self.db = db_manager
        self.patient_id = patient_id
        
        # Initialize face recognition with trained model
        try:
            self.face_recognition = FaceRecognitionSystem('hardware/models/')
            self.logger.info("Face recognition system loaded")
        except Exception as e:
            self.logger.error(f"Failed to load face recognition model: {e}")
            self.face_recognition = None
        
        # Load known persons from database
        self._sync_known_persons()
    
    def _sync_known_persons(self):
        """Sync known persons from database to face recognition system"""
        try:
            known_persons = self.db.get_all_known_persons(self.patient_id)
            self.logger.info(f"Loaded {len(known_persons)} known persons from database")
        except Exception as e:
            self.logger.error(f"Failed to sync known persons: {e}")
    
    async def identify_person(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Identify person in frame and return their information"""
        try:
            if self.face_recognition is None:
                return None
            
            # Detect faces using FaceRecognitionSystem
            face_detections = self.face_recognition.detect_faces(frame)
            
            if not face_detections or len(face_detections) == 0:
                return None
            
            # Get the first detected face
            face_detection = face_detections[0]
            face_image = face_detection.get('face_image')
            
            if face_image is None:
                return None
            
            # Recognize person
            recognition_result = self.face_recognition.recognize_person(face_image)
            
            if recognition_result and recognition_result.get('name'):
                name = recognition_result.get('name')
                confidence = recognition_result.get('confidence', 0.0)
                
                # Query database for person details
                all_persons = self.db.get_all_known_persons(self.patient_id)
                db_person = next((p for p in all_persons if p['name'] == name), None)
                
                if db_person:
                    # Update recognition count
                    self.db.update_person_recognition(db_person['id'])
                    
                    return {
                        'name': db_person['name'],
                        'relationship': db_person['relationship'],
                        'confidence': confidence,
                        'face_bbox': face_detection.get('bbox'),
                        'last_seen': db_person.get('last_seen'),
                        'recognition_count': db_person.get('recognition_count', 0) + 1
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Person identification failed: {e}")
            return None
    
    def add_known_person(self, name: str, relationship: str, 
                        face_images: list, photo_path: str = None) -> bool:
        """Add a new known person to the system"""
        try:
            if self.face_recognition is None:
                self.logger.error("Face recognition not initialized")
                return False
            
            # Register person with face recognition system
            success = self.face_recognition.add_person(name, relationship, face_images)
            
            if success:
                # Add to database
                person_id = self.db.add_known_person(
                    patient_id=self.patient_id,
                    name=name,
                    relationship=relationship,
                    photo_path=photo_path
                )
                
                if person_id > 0:
                    self.logger.info(f"Successfully added {name} as known person")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add known person {name}: {e}")
            return False