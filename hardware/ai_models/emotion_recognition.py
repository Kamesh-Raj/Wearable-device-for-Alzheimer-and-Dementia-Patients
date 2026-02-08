"""
Emotion Recognition - Optimized for Raspberry Pi Zero 2W
Uses TFLite for fast inference.
Supports personal calibration (fine-tuning) for patient baseline.
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import logging
import os
import json

class EmotionRecognizer:
    def __init__(self, model_path="hardware/models/emotion_model.tflite"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Personal calibration data
        self.calibration_file = "hardware/models/emotion_calibration.json"
        self.baseline_offsets = {label: 0.0 for label in self.labels}
        
        self.load_model()
        self.load_calibration()

    def load_model(self):
        """Load TFLite model"""
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}. Please download/train first.")
                return

            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.logger.info("Emotion model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load emotion model: {e}")

    def load_calibration(self):
        """Load patient-specific baseline offsets"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.baseline_offsets = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load calibration: {e}")

    def calibrate(self, face_images, expected_emotion="Neutral"):
        """
        Personal Fine-Tuning / Calibration:
        Adjust model sensitivity based on patient's baseline expressions.
        E.g., if patient naturally looks 'Sad', we adjust the Neutral threshold.
        """
        if not self.interpreter:
            return
            
        avg_scores = {label: 0.0 for label in self.labels}
        count = 0 
        
        for img in face_images:
            scores = self._raw_predict(img)
            if scores:
                for label, score in scores.items():
                    avg_scores[label] += score
                count += 1
        
        if count == 0:
            return

        # Normalize
        for label in avg_scores:
            avg_scores[label] /= count

        # Calculate offsets
        # If model detects 'Sad' (0.8) but user says it's 'Neutral',
        # we learn that this user has a resting 'Sad' face.
        # Simple Logic: Bias the expected emotion upwards, others downwards.
        target_bias = 0.2  # Slight nudge
        
        for label in self.labels:
            current_avg = avg_scores[label]
            if label == expected_emotion:
                # If prediction is lower than expected, add positive offset
                self.baseline_offsets[label] = target_bias
            else:
                # If prediction is higher than it should be (false positive), add negative offset
                if current_avg > 0.3:
                     self.baseline_offsets[label] = -0.1

        # Save calibration
        with open(self.calibration_file, 'w') as f:
            json.dump(self.baseline_offsets, f)
        self.logger.info(f"Calibration updated for emotion baseline: {expected_emotion}")

    def _raw_predict(self, face_image):
        """Run inference without calibration"""
        if not self.interpreter:
            return None

        # Preprocess
        input_shape = self.input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])
        
        resized = cv2.resize(face_image, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        scores = output_data[0]
        result = {label: float(score) for label, score in zip(self.labels, scores)}
        return result

    def predict(self, face_image):
        """
        Predict emotion from face image
        Returns: (emotion_label, confidence)
        """
        raw_scores = self._raw_predict(face_image)
        if not raw_scores:
            return "Unknown", 0.0

        # Apply calibration offsets
        adjusted_scores = {}
        for label, score in raw_scores.items():
            adjusted_scores[label] = score + self.baseline_offsets.get(label, 0.0)

        # Get max
        best_label = max(adjusted_scores, key=adjusted_scores.get)
        confidence = adjusted_scores[best_label]
        
        return best_label, confidence