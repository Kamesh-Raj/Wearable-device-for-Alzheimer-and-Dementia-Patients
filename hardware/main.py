"""
Main Orchestration Script for Raspberry Pi Hardware
Runs the neuro-assistive system loop.
"""

import time
import threading
import logging
import cv2
import config
from modules.camera_module import CameraModule
from modules.audio_module import AudioModule
from modules.display_module import DisplayModule
from modules.haptic_module import HapticModule
from modules.gps_module import GPSModule
from ai_models.face_model import FaceRecognizer
from ai_models.emotion_recognition import EmotionRecognizer
from ai_models.assistive_ai import AssistiveAI
from ai_models.multimodal_fusion import MultimodalFusion
from alert_manager import AlertManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroAssistHardware:
    def __init__(self):
        logger.info("Initializing Hardware System...")
        
        # Hardware Modules
        self.camera = CameraModule(rotation=config.CAMERA_ROTATION, resolution=config.CAMERA_RESOLUTION)
        self.audio = AudioModule()
        self.display = DisplayModule()
        self.haptic = HapticModule(pin=config.HAPTIC_PIN)
        self.gps = GPSModule()
        
        # AI Models
        self.face_ai = FaceRecognizer(model_dir=config.FACE_MODEL_DIR)
        self.emotion_ai = EmotionRecognizer(model_path=config.EMOTION_MODEL_PATH)
        self.assistive_ai = AssistiveAI()
        self.fusion_ai = MultimodalFusion()
        
        # Backend Connection
        self.alert_manager = AlertManager(api_base_url=config.API_BASE_URL, patient_id=config.PATIENT_ID)
        
        self.running = True
        self.current_emotion = "Neutral"
        self.last_voice_command_time = 0

    def start(self):
        """Start main loops"""
        # Start Audio Thread
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        
        # Start Alert/Reminder Polling Thread
        self.alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self.alert_thread.start()
        
        # Main Camera Loop (Blocking)
        self._camera_loop()

    def _camera_loop(self):
        """Process video frames for face/emotion recognition"""
        logger.info("Starting Camera Loop...")
        
        while self.running:
            frame = self.camera.get_frame(auto_rotate=True)
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Recognize Faces
            faces = self.face_ai.recognize(frame)
            
            if not faces:
                self.display.clear()
                self.display.show_text("Scanning...")
            
            for face in faces:
                name = face['name']
                user_type = face['type']
                confidence = face['confidence']
                
                # Check Identify
                if user_type == 'patient':
                    # Analyze Emotion for Patient
                    # Crop face from frame using location
                    top, right, bottom, left = face['location']
                    face_img = frame[top:bottom, left:right]
                    
                    if face_img.size > 0:
                        emotion, conf = self.emotion_ai.predict(face_img)
                        self.current_emotion = emotion
                        
                        # Update OLED with calming emoji if needed
                        self.display.show_emoji(emotion)
                        
                        # Check for Distress
                        if emotion in ["Fear", "Sad", "Angry"] and conf > 0.7:
                             if time.time() - self.last_voice_command_time > 10: # Don't spam
                                response, _ = self.assistive_ai.get_response("calm")
                                self.audio.speak(response)
                                self.last_voice_command_time = time.time()
                                
                                # Send Alert
                                self.alert_manager.send_alert("emotional_distress", f"Patient shows intense {emotion}", severity="medium")

                elif user_type == 'known' or user_type == 'caregiver':
                    # Recognized Person
                    logger.info(f"Recognized: {name}")
                    self.display.show_text(f"Hello, {name}")
                    # Audio greeting (once per encounter logic omitted for brevity)
                    # self.audio.speak(f"Hello {name}")
                    
                else:
                    # Unknown
                    self.display.show_text("Unknown Person")

            time.sleep(0.1) # FPS limit for Pi

    def _audio_loop(self):
        """Listen for voice commands"""
        logger.info("Starting Audio Loop...")
        for text in self.audio.listen_continuously():
            if not self.running: break
            
            if text:
                logger.info(f"Heard: {text}")
                response, action_required, intent = self.assistive_ai.process_voice_input(text)
                
                # Update timestamp to avoid conflicting with visual emotion checks
                self.last_voice_command_time = time.time()
                
                # Speak response
                self.audio.speak(response)
                
                # Update Display
                self.display.show_text(response[:20]) # Show brief text
                
                 # Map intent to emotion for fusion
                voice_emotion_map = {
                    "help": "Fear",
                    "calm": "Sad",
                    "happy": "Happy"
                }
                voice_emotion = voice_emotion_map.get(intent, "Neutral")
                
                # Fuse with current visual emotion (accessed from shared state)
                visual_result = {'emotion': self.current_emotion, 'confidence': 0.8} # Use real confidence if stored
                voice_result = {'emotion': voice_emotion, 'confidence': 0.9 if intent != 'unknown' else 0.3}
                
                final_emotion, conf, meta = self.fusion_ai.fuse(visual_result, voice_result)
                logger.info(f"Fused Emotion: {final_emotion} ({conf:.2f})")
                
                # Handle Actions
                if action_required or (final_emotion in ["Fear", "Sad"] and conf > 0.7):
                    if intent == "help" or final_emotion == "Fear":
                        self.alert_manager.send_alert("voice_help", f"Patient Distress: {final_emotion}", severity="high")
                        self.haptic.alert_pattern()

    def _alert_loop(self):
        """Poll for reminders and check safety zones"""
        logger.info("Starting Alert Loop...")
        while self.running:
            # Check Reminders
            reminders = self.alert_manager.get_reminders()
            for rem in reminders:
                # If reminder is due (backend handles logic usually, but here simplicity)
                msg = f"Reminder: {rem['title']}"
                self.display.show_text(msg)
                self.audio.speak(msg)
                self.haptic.vibrate()
                time.sleep(5) # Wait before processing next
            
            # Check Safe Zone (Mock GPS for now)
            # In production, use GPS module to get lat/lon
            # Check Safe Zone
            current_lat, current_lon = self.gps.get_location() or (0.0, 0.0)
            is_safe, msg = self.alert_manager.check_safe_zone(current_lat, current_lon)
            
            if not is_safe:
                self.alert_manager.send_alert("safe_zone_violation", msg, severity="high")
                self.audio.speak("Warning. You are leaving the safe zone.")
                self.display.show_emoji("alert")
                self.haptic.alert_pattern()
            
            time.sleep(30) # Poll every 30s

    def stop(self):
        self.running = False
        if self.camera: self.camera.release()
        if self.gps: self.gps.stop()
        logger.info("System Stopped.")

if __name__ == "__main__":
    app = NeuroAssistHardware()
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()