"""
Configuration for Hardware Modules
"""

# API Settings
API_BASE_URL = "http://127.0.0.1:5000/api"
PATIENT_ID = 1

# Hardware Pins (GPIO)
HAPTIC_PIN = 17
BUTTON_PIN = 27

# Camera Settings
CAMERA_ROTATION = 0 # 0, 90, 180, 270
CAMERA_RESOLUTION = (640, 480)

# Audio Settings
MIC_DEVICE_INDEX = 1 # Check `arecord -l`
SPEAKER_DEVICE_INDEX = 0

# Display Settings
OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_ADDRESS = 0x3C

# Model Paths
FACE_MODEL_DIR = "hardware/models/face_recognition"
EMOTION_MODEL_PATH = "hardware/models/emotion_model.tflite"
VOSK_MODEL_PATH = "hardware/models/vosk-model-small-en-us-0.15"

# Safe Zone Check Interval (seconds)
SAFE_ZONE_INTERVAL = 300
