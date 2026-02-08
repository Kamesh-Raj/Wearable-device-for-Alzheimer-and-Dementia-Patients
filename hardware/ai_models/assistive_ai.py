"""
Assistive AI - Voice Assistant Logic
Optimized for Raspberry Pi Zero 2W (low memory usage).
Uses Intent Recognition for commands and simple conversation.
"""

import logging
import datetime
import random

class AssistiveAI:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Local intent database (Simple but effective for Pi Zero)
        self.intents = {
            "time": ["what time is it", "tell me the time", "current time"],
            "date": ["what is today", "what day is it", "current date"],
            "location": ["where am i", "what is this place", "lost"],
            "medication": ["did i take my pills", "medication", "pills"],
            "family": ["call my daughter", "call son", "who are you"], 
            "help": ["help me", "emergency", "i need help"],
            "calm": ["i am scared", "anxious", "worried"]
        }
        
        self.responses = {
            "default": ["I'm here to help you.", "Can you repeat that?", "I am listening."],
            "calm": ["Take a deep breath. You are safe.", "Everything is okay.", "Let's listen to some music."],
            "location": ["You are at home, in the living room.", "You are safe at home."],
            "medication": ["Let me check your schedule. Yes, you took your morning pills."], # Hardcoded for now, would fetch from Reminder API
            "family": ["I will notify your family.", "Calling your emergency contact."] # Initiates action
        }

    def robust_intent_recognition(self, text):
        """
        Process transcribed text and identify intent.
        
        Args:
            text: Transcribed string from STT engine
            
        Returns:
            intent: Detected intent key or 'unknown'
            confidence: float (0.0 - 1.0)
        """
        text = text.lower()
        best_intent = "unknown"
        max_score = 0
        
        for intent, keywords in self.intents.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            
            if score > max_score:
                max_score = score
                best_intent = intent
                
        return best_intent, max_score > 0

    def get_response(self, intent, confidence=1.0):
        """
        Generate response based on intent.
        
        Args:
            intent: Detected intent string
            
        Returns:
            response_text: String to be spoken
            action_required: boolean
        """
        if confidence < 0.5:
             return random.choice(self.responses["default"]), False

        if intent == "time":
            now = datetime.datetime.now()
            return f"It is {now.strftime('%I:%M %p')}.", False
            
        elif intent == "date":
            today = datetime.datetime.now()
            return f"Today is {today.strftime('%A, %B %d')}.", False
            
        elif intent == "help":
            return "I am alerting your caregiver immediately. Stay calm.", True # Action: Trigger Alert
        
        elif intent in self.responses:
            return random.choice(self.responses[intent]), False # Simple response
            
        return "I'm not sure I understand.", False

    def process_voice_input(self, text):
        """
        Main pipeline: Speech Text -> Intent -> Response
        """
        self.logger.info(f"Processing voice input: {text}")
        intent, valid = self.robust_intent_recognition(text)
        
        if valid:
            response, action = self.get_response(intent)
            self.logger.info(f"Detected intent: {intent}, Response: {response}")
            return response, action, intent
        else:
            return self.get_response("default")[0], False, "unknown"
