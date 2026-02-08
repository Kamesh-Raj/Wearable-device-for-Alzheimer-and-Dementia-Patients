"""
Emotion-Responsive AI Module
Integrates emotion recognition with ChatGPT API for personalized support
Uses trained models: face_emotion_95_int8.tflite, voice_emotion_optimized_int8.tflite
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from hardware.ai_models.assistive_ai import AssistiveAI

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available - using fallback responses")

class EmotionAI:
    """Emotion-responsive AI system for patient support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize the complete assistive AI system
        try:
            self.assistive_system = AssistiveAI()
            self.logger.info("Assistive AI system initialized with trained models")
        except Exception as e:
            self.logger.error(f"Failed to initialize assistive AI system: {e}")
            self.assistive_system = None
        
        # OpenAI configuration
        self.openai_client = None
        if OPENAI_AVAILABLE:
            self._setup_openai()
        
        # Fallback responses for different emotional states
        self.fallback_responses = {
            'angry': [
                "I understand you might be feeling frustrated. Let's take a deep breath together.",
                "It's okay to feel upset sometimes. Would you like to sit down for a moment?",
                "I'm here with you. Let's try to relax and take things slowly."
            ],
            'sad': [
                "I can see you might be feeling down. You're not alone - I'm here with you.",
                "It's natural to feel sad sometimes. Would you like to listen to some calming music?",
                "You're doing great. Remember that difficult feelings will pass."
            ],
            'fear': [
                "You're safe right now. I'm here to help you feel more secure.",
                "It's okay to feel worried. Let's focus on staying calm and breathing slowly.",
                "You're in a safe place. Everything is going to be alright."
            ],
            'happy': [
                "It's wonderful to see you feeling good! Your smile brightens the day.",
                "I'm so glad you're feeling happy. Keep enjoying this positive moment.",
                "Your happiness is contagious! Thank you for sharing this joy."
            ],
            'neutral': [
                "How are you feeling today? I'm here if you need anything.",
                "You seem calm and peaceful. Is there anything I can help you with?",
                "I'm here to support you. Let me know if you'd like to chat or need assistance."
            ]
        }
    
    def _setup_openai(self):
        """Setup OpenAI client if API key is available"""
        try:
            # In a real implementation, this would come from environment variables
            # or device settings
            api_key = "your-openai-api-key"  # Replace with actual key
            if api_key and api_key != "your-openai-api-key":
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.logger.info("OpenAI client initialized")
            else:
                self.logger.warning("OpenAI API key not configured - using fallback responses")
        except Exception as e:
            self.logger.error(f"OpenAI setup failed: {e}")
    
    async def analyze_emotion(self, frame: np.ndarray, 
                            audio_data: np.ndarray = None) -> Dict[str, Any]:
        """Analyze patient's emotional state from visual and audio cues using trained models"""
        try:
            if self.assistive_system is None:
                return {
                    'visual_emotion': None,
                    'audio_emotion': None,
                    'distress_level': 0.0,
                    'error': 'assistive_system_not_initialized'
                }
            
            # Process through assistive AI system
            results = self.assistive_system.process_frame(frame, audio_data)
            
            # Extract emotion information
            emotion_result = {
                'visual_emotion': results.get('emotion'),
                'audio_emotion': results.get('voice_emotion'),
                'distress_level': self.assistive_system.patient_state.stress_level,
                'current_emotion': self.assistive_system.patient_state.emotion,
                'emotion_confidence': self.assistive_system.patient_state.emotion_confidence,
                'needs_attention': self.assistive_system.patient_state.needs_attention,
                'recommendations': results.get('recommendations', [])
            }
            
            # Log the emotion data
            self.logger.info(f"Emotion: {emotion_result['current_emotion']} "
                           f"({emotion_result['emotion_confidence']:.2f}), "
                           f"Distress: {emotion_result['distress_level']:.2f}")
            
            return emotion_result
            
        except Exception as e:
            self.logger.error(f"Emotion analysis failed: {e}")
            return {
                'visual_emotion': None,
                'audio_emotion': None,
                'distress_level': 0.0,
                'error': str(e)
            }
    
    async def generate_support_message(self, emotion_state: Dict[str, Any], 
                                      user_text: str = "") -> str:
        """Generate personalized emotional support message"""
        try:
            distress_level = emotion_state.get('distress_level', 0.0)
            current_emotion = emotion_state.get('current_emotion', 'neutral')
            
            # Use OpenAI if available and configured
            if self.openai_client and distress_level > 0.5:
                return await self._generate_ai_response(current_emotion, distress_level, emotion_state)
            else:
                # Use fallback responses
                return self._get_fallback_response(current_emotion, distress_level)
                
        except Exception as e:
            self.logger.error(f"Support message generation failed: {e}")
            return "I'm here with you. You're doing great, and I'm here to help."
    
    def _get_dominant_emotion(self, emotion_state: Dict[str, Any]) -> str:
        """Determine the dominant emotion from the analysis"""
        try:
            visual_emotion = emotion_state.get('visual_emotion')
            audio_emotion = emotion_state.get('audio_emotion')
            
            # Prioritize visual emotion if available
            if visual_emotion and visual_emotion.get('dominant_emotion'):
                return visual_emotion['dominant_emotion']
            elif audio_emotion and audio_emotion.get('dominant_emotion'):
                return audio_emotion['dominant_emotion']
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Failed to determine dominant emotion: {e}")
            return 'neutral'
    
    async def _generate_ai_response(self, dominant_emotion: str, distress_level: float,
                                  emotion_state: Dict[str, Any]) -> str:
        """Generate AI response using ChatGPT"""
        try:
            # Create context-aware prompt
            prompt = self._create_support_prompt(dominant_emotion, distress_level, emotion_state)
            
            # Call OpenAI API
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a compassionate AI assistant helping an elderly person with Alzheimer's. Provide brief, calming, and supportive responses. Keep responses under 50 words and use simple, clear language."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            ai_message = response.choices[0].message.content.strip()
            
            # Validate response length and content
            if len(ai_message) > 200:  # Too long
                return self._get_fallback_response(dominant_emotion, distress_level)
            
            return ai_message
            
        except Exception as e:
            self.logger.error(f"AI response generation failed: {e}")
            return self._get_fallback_response(dominant_emotion, distress_level)
    
    def _create_support_prompt(self, dominant_emotion: str, distress_level: float,
                             emotion_state: Dict[str, Any]) -> str:
        """Create context-aware prompt for AI response"""
        prompt = f"The person is showing signs of {dominant_emotion} emotion"
        
        if distress_level > 0.7:
            prompt += " and appears to be in significant distress"
        elif distress_level > 0.4:
            prompt += " and seems somewhat upset"
        else:
            prompt += " and appears relatively calm"
        
        prompt += ". Please provide a brief, comforting response that acknowledges their feelings and offers gentle support."
        
        # Add specific context based on emotion
        emotion_contexts = {
            'angry': " They seem frustrated or agitated.",
            'sad': " They appear to be feeling down or melancholy.",
            'fear': " They seem worried or anxious about something.",
            'happy': " They're showing positive emotions.",
            'surprise': " They seem startled or surprised.",
            'disgust': " They appear uncomfortable or displeased.",
            'neutral': " They seem calm but might need gentle engagement."
        }
        
        prompt += emotion_contexts.get(dominant_emotion, "")
        
        return prompt
    
    def _get_fallback_response(self, dominant_emotion: str, distress_level: float) -> str:
        """Get appropriate fallback response"""
        try:
            import random
            
            responses = self.fallback_responses.get(dominant_emotion, self.fallback_responses['neutral'])
            
            # Adjust response based on distress level
            if distress_level > 0.8:
                # High distress - use most calming responses
                calming_responses = [
                    "You're safe right now. I'm here with you, and everything is going to be okay.",
                    "Let's take this one moment at a time. You're doing great, and I'm here to help.",
                    "I understand this feels overwhelming. Let's breathe together and take it slowly."
                ]
                return random.choice(calming_responses)
            elif distress_level > 0.5:
                # Moderate distress - use supportive responses
                return random.choice(responses)
            else:
                # Low distress - use gentle, positive responses
                positive_responses = [
                    "You're doing wonderfully today. I'm here if you need anything.",
                    "It's nice to spend this time with you. How are you feeling?",
                    "You seem peaceful right now. I'm here to support you."
                ]
                return random.choice(positive_responses)
                
        except Exception as e:
            self.logger.error(f"Fallback response selection failed: {e}")
            return "I'm here with you. You're doing great."
    
    def get_emotion_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of recent emotional patterns"""
        # In a real implementation, this would query the database
        # for recent emotion logs and provide analytics
        return {
            'status': 'not_implemented',
            'message': 'Emotion summary requires database integration'
        }