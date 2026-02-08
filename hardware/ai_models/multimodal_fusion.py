"""
Multimodal Fusion - Combine Face and Voice Emotion Analysis
Handles conflict resolution, confidence weighting, and final decision making.
"""

import logging

class MultimodalFusion:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Weights for modalities based on reliability
        self.face_weight = 0.6  # Face is generally more reliable for continuous monitoring
        self.voice_weight = 0.4
        self.conflict_threshold = 0.4 # Significant disagreement

    def fuse(self, face_result, voice_result):
        """
        Combine emotion predictions from face and voice.
        
        Args:
            face_result: dict {'emotion': str, 'confidence': float} or None
            voice_result: dict {'emotion': str, 'confidence': float} or None
            
        Returns:
            final_emotion: str
            confidence: float
            metadata: dict (conflict detected, dominant modality)
        """
        
        if not face_result and not voice_result:
            return "Neutral", 0.0, {}

        # Fallback to single modality if one is missing
        if not face_result:
            return voice_result['emotion'], voice_result['confidence'], {'source': 'voice_only'}
        
        if not voice_result:
            return face_result['emotion'], face_result['confidence'], {'source': 'face_only'}

        # Both Available - Perform Weighted Average Fusion
        face_conf = face_result.get('confidence', 0.5)
        voice_conf = voice_result.get('confidence', 0.5)
        
        # Calculate scores for each emotion (assuming standardized emotion set)
        # Here we simplify: Just compare top predictions
        
        face_emotion = face_result['emotion']
        voice_emotion = voice_result['emotion']
        
        if face_emotion == voice_emotion:
            # Agreement: Boost confidence
            combined_conf = min(1.0, (face_conf * self.face_weight) + (voice_conf * self.voice_weight) + 0.1)
            return face_emotion, combined_conf, {'source': 'agreement'}
        
        else:
            # Disagreement: Check confidence
            weighted_face = face_conf * self.face_weight
            weighted_voice = voice_conf * self.voice_weight
            
            conflict_score = abs(weighted_face - weighted_voice)
            
            if weighted_face > weighted_voice:
                final_emotion = face_emotion
                final_conf = weighted_face
                dominant = "face"
            else:
                final_emotion = voice_emotion
                final_conf = weighted_voice
                dominant = "voice"
                
            # Detect serious conflict (e.g., Happy vs Angry)
            # This logic could be expanded with an emotion distance matrix
            is_conflict = conflict_score > self.conflict_threshold
            
            return final_emotion, final_conf, {'source': 'weighted', 'conflict': is_conflict, 'dominant': dominant}

    def assess_risk(self, emotion, gait_anomaly=False):
        """
        Determine if immediate alert is needed based on fused emotion + gait.
        """
        high_risk_emotions = ['Fear', 'Distress', 'Pain', 'Angry']
        
        risk_level = "Low"
        
        if emotion in high_risk_emotions:
            risk_level = "Medium"
            
        if gait_anomaly:
            if risk_level == "Medium":
                risk_level = "Critical"
            else:
                risk_level = "High"
                
        return risk_level
