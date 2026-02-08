import os
import shutil
import logging
from .face_model import FaceRecognizer # Use relative import if package, or face_model if script
try:
    from face_model import FaceRecognizer
except ImportError:
    # Fallback for relative run
    from .face_model import FaceRecognizer

# Paths
BACKEND_UPLOADS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/static/uploads"))
TRAINING_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../training_data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/face_recognition"))

# Source Directories in Backend
SRC_PATIENT = os.path.join(BACKEND_UPLOADS, "profiles")
SRC_KNOWN = os.path.join(BACKEND_UPLOADS, "known_persons")
SRC_CAREGIVER = os.path.join(BACKEND_UPLOADS, "caregivers") # Logical placeholder

# Destination Directories in Hardware
DEST_PATIENT = os.path.join(TRAINING_DATA_DIR, "patient")
DEST_KNOWN = os.path.join(TRAINING_DATA_DIR, "known")
DEST_CAREGIVER = os.path.join(TRAINING_DATA_DIR, "caregiver")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_images():
    """Copy new images from backend to training folder"""
    logger.info("Syncing images...")
    
    # 1. Patient Images
    if os.path.exists(SRC_PATIENT):
        os.makedirs(DEST_PATIENT, exist_ok=True)
        # Walk through patient folders 
        # Structure: uploads/profiles/<id>/photo.jpg
        for patient_id in os.listdir(SRC_PATIENT):
            patient_path = os.path.join(SRC_PATIENT, patient_id)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        src = os.path.join(patient_path, file)
                        # Rename to include ID for uniqueness: patient_<id>_<filename>
                        dest_name = f"patient_{patient_id}_{file}"
                        dst = os.path.join(DEST_PATIENT, dest_name)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            logger.info(f"Copied {file} to training data.")

    # 2. Known Persons Images
    if os.path.exists(SRC_KNOWN):
        os.makedirs(DEST_KNOWN, exist_ok=True)
         # Structure: uploads/known_persons/<patient_id>/filename.jpg
        for patient_id in os.listdir(SRC_KNOWN):
             patient_known_path = os.path.join(SRC_KNOWN, patient_id)
             if os.path.isdir(patient_known_path):
                 for file in os.listdir(patient_known_path):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        src = os.path.join(patient_known_path, file)
                        # Name usually contains name_timestamp_filename from app.py
                        dest_name = f"known_{patient_id}_{file}"
                        dst = os.path.join(DEST_KNOWN, dest_name)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            logger.info(f"Copied {file} to training data.")
    
    # Caregivers - similar logic if implemented

def run_fine_tuning():
    """Execute the fine-tuning process"""
    sync_images()
    
    recognizer = FaceRecognizer(model_dir=MODEL_DIR)
    
    # Point to the training data directory relative to this script
    recognizer.fine_tune(data_dir=TRAINING_DATA_DIR)
    
    logger.info(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    run_fine_tuning()