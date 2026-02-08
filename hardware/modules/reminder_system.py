"""
Triple-Mode Reminder System
Handles medication and appointment reminders with OLED, Audio, and Haptic feedback
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

class ReminderSystem:
    """Manages medication and appointment reminders with multi-modal alerts"""
    
    def __init__(self, db_manager, patient_id: int):
        self.logger = logging.getLogger(__name__)
        self.db = db_manager
        self.patient_id = patient_id
        
        # Reminder configuration
        self.reminder_types = {
            'medication': {
                'icon': '💊',
                'priority': 'high',
                'repeat_alert_minutes': 5
            },
            'appointment': {
                'icon': '📅',
                'priority': 'medium',
                'repeat_alert_minutes': 15
            },
            'activity': {
                'icon': '🏃',
                'priority': 'low',
                'repeat_alert_minutes': 30
            }
        }
    
    async def check_due_reminders(self) -> List[Dict[str, Any]]:
        """Check for reminders that are due now"""
        try:
            current_time = datetime.now()
            due_reminders = self.db.get_due_reminders(self.patient_id, current_time)
            
            # Process each due reminder
            processed_reminders = []
            for reminder in due_reminders:
                processed_reminder = self._prepare_reminder_message(reminder)
                processed_reminders.append(processed_reminder)
            
            return processed_reminders
            
        except Exception as e:
            self.logger.error(f"Failed to check due reminders: {e}")
            return []
    
    def _prepare_reminder_message(self, reminder: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare reminder message with appropriate formatting"""
        try:
            reminder_type = reminder['type']
            config = self.reminder_types.get(reminder_type, self.reminder_types['activity'])
            
            # Create display message
            icon = config['icon']
            title = reminder['title']
            description = reminder.get('description', '')
            
            display_message = f"{icon} {title}"
            if description:
                display_message += f"\n{description}"
            
            # Create audio message
            audio_message = f"Reminder: {title}"
            if description:
                audio_message += f". {description}"
            
            # Add time context
            scheduled_time = datetime.fromisoformat(reminder['scheduled_time'])
            if scheduled_time.date() == datetime.now().date():
                time_str = scheduled_time.strftime("%I:%M %p")
                audio_message += f" scheduled for {time_str}"
            
            return {
                'id': reminder['id'],
                'type': reminder_type,
                'priority': config['priority'],
                'display_message': display_message,
                'audio_message': audio_message,
                'icon': icon,
                'scheduled_time': reminder['scheduled_time'],
                'repeat_alert_minutes': config['repeat_alert_minutes']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare reminder message: {e}")
            return {
                'id': reminder.get('id', -1),
                'type': 'unknown',
                'priority': 'low',
                'display_message': 'Reminder',
                'audio_message': 'You have a reminder',
                'icon': '⏰'
            }
    
    async def trigger_reminder(self, reminder: Dict[str, Any], sensors) -> bool:
        """Trigger triple-mode reminder (OLED + Audio + Haptic)"""
        try:
            reminder_id = reminder['id']
            priority = reminder.get('priority', 'medium')
            
            # Determine alert intensity based on priority
            if priority == 'high':
                vibration_pattern = 'strong'
                audio_volume = 'loud'
                display_duration = 30  # seconds
            elif priority == 'medium':
                vibration_pattern = 'medium'
                audio_volume = 'normal'
                display_duration = 20
            else:
                vibration_pattern = 'gentle'
                audio_volume = 'soft'
                display_duration = 15
            
            # Execute triple-mode alert simultaneously
            await asyncio.gather(
                self._display_reminder(reminder, sensors, display_duration),
                self._announce_reminder(reminder, sensors, audio_volume),
                self._vibrate_reminder(reminder, sensors, vibration_pattern)
            )
            
            # Mark reminder as triggered in database
            self.db.mark_reminder_triggered(reminder_id)
            
            # Log the reminder event
            self.db.log_patient_data(
                patient_id=self.patient_id,
                log_type='interaction',
                data={
                    'event': 'reminder_triggered',
                    'reminder_type': reminder['type'],
                    'reminder_title': reminder.get('display_message', ''),
                    'priority': priority
                },
                severity_level=1 if priority == 'high' else 0
            )
            
            self.logger.info(f"Triggered {priority} priority reminder: {reminder.get('display_message', '')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to trigger reminder: {e}")
            return False
    
    async def _display_reminder(self, reminder: Dict[str, Any], sensors, duration: int):
        """Display reminder on OLED screen"""
        try:
            message = reminder['display_message']
            
            # Display the message
            await sensors.display_message(message)
            
            # Keep message displayed for specified duration
            await asyncio.sleep(duration)
            
            # Clear display (or show default screen)
            await sensors.display_message("")
            
        except Exception as e:
            self.logger.error(f"Failed to display reminder: {e}")
    
    async def _announce_reminder(self, reminder: Dict[str, Any], sensors, volume: str):
        """Announce reminder through speaker"""
        try:
            message = reminder['audio_message']
            
            # Adjust message for speech synthesis
            speech_message = message.replace('💊', 'medication')
            speech_message = speech_message.replace('📅', 'appointment')
            speech_message = speech_message.replace('🏃', 'activity')
            
            # Speak the reminder
            await sensors.speak(speech_message)
            
            # For high priority reminders, repeat after a short pause
            if reminder.get('priority') == 'high':
                await asyncio.sleep(3)
                await sensors.speak("This is an important reminder.")
            
        except Exception as e:
            self.logger.error(f"Failed to announce reminder: {e}")
    
    async def _vibrate_reminder(self, reminder: Dict[str, Any], sensors, pattern: str):
        """Provide haptic feedback for reminder"""
        try:
            if pattern == 'strong':
                # Strong vibration pattern for high priority
                for _ in range(3):
                    await sensors.vibrate_alert()
                    await asyncio.sleep(0.5)
            elif pattern == 'medium':
                # Medium vibration pattern
                for _ in range(2):
                    await sensors.vibrate_alert()
                    await asyncio.sleep(0.3)
            else:
                # Gentle vibration pattern
                await sensors.vibrate_gentle()
            
        except Exception as e:
            self.logger.error(f"Failed to vibrate reminder: {e}")
    
    def add_medication_reminder(self, medication_name: str, dosage: str,
                              times: List[str], start_date: datetime = None) -> List[int]:
        """Add medication reminders for specified times"""
        try:
            if start_date is None:
                start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            reminder_ids = []
            
            for time_str in times:
                # Parse time (e.g., "08:00", "14:30")
                hour, minute = map(int, time_str.split(':'))
                reminder_time = start_date.replace(hour=hour, minute=minute)
                
                # If time has passed today, schedule for tomorrow
                if reminder_time <= datetime.now():
                    reminder_time += timedelta(days=1)
                
                # Create reminder
                title = f"{medication_name}"
                description = f"Take {dosage}"
                
                repeat_pattern = {
                    'type': 'daily',
                    'interval': 1
                }
                
                reminder_id = self.db.add_reminder(
                    patient_id=self.patient_id,
                    reminder_type='medication',
                    title=title,
                    description=description,
                    scheduled_time=reminder_time,
                    repeat_pattern=repeat_pattern
                )
                
                if reminder_id > 0:
                    reminder_ids.append(reminder_id)
            
            self.logger.info(f"Added {len(reminder_ids)} medication reminders for {medication_name}")
            return reminder_ids
            
        except Exception as e:
            self.logger.error(f"Failed to add medication reminder: {e}")
            return []
    
    def add_appointment_reminder(self, appointment_title: str, appointment_time: datetime,
                               advance_notice_minutes: int = 30) -> int:
        """Add appointment reminder with advance notice"""
        try:
            # Calculate reminder time (before appointment)
            reminder_time = appointment_time - timedelta(minutes=advance_notice_minutes)
            
            # Create reminder
            description = f"Appointment at {appointment_time.strftime('%I:%M %p')}"
            
            reminder_id = self.db.add_reminder(
                patient_id=self.patient_id,
                reminder_type='appointment',
                title=appointment_title,
                description=description,
                scheduled_time=reminder_time
            )
            
            if reminder_id > 0:
                self.logger.info(f"Added appointment reminder: {appointment_title}")
            
            return reminder_id
            
        except Exception as e:
            self.logger.error(f"Failed to add appointment reminder: {e}")
            return -1
    
    def get_upcoming_reminders(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming reminders within specified time window"""
        try:
            # This would query the database for upcoming reminders
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get upcoming reminders: {e}")
            return []
    
    def cancel_reminder(self, reminder_id: int) -> bool:
        """Cancel a specific reminder"""
        try:
            # Deactivate reminder in database
            cursor = self.db.connection.cursor()
            cursor.execute("UPDATE reminders SET is_active = 0 WHERE id = ?", (reminder_id,))
            self.db.connection.commit()
            
            success = cursor.rowcount > 0
            if success:
                self.logger.info(f"Cancelled reminder {reminder_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to cancel reminder: {e}")
            return False