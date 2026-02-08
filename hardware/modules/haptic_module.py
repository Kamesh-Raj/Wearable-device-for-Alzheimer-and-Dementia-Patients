"""
Haptic Module - Controls Vibration Motor for Alerts
"""

import time
import logging
# Use gpiozero or RPi.GPIO
try:
    from gpiozero import PWMOutputDevice
except ImportError:
    PWMOutputDevice = None # Mock if not on Pi

class HapticModule:
    def __init__(self, pin=17):
        self.logger = logging.getLogger(__name__)
        self.device = None
        if PWMOutputDevice:
            try:
                self.device = PWMOutputDevice(pin)
                self.logger.info("Haptic motor initialized.")
            except Exception as e:
                self.logger.error(f"Failed to init haptic: {e}")
        else:
            self.logger.warning("GPIO library not found. Haptic disabled.")

    def vibrate(self, duration=0.5, intensity=0.8):
        """Standard vibration"""
        if self.device:
            self.device.value = intensity
            time.sleep(duration)
            self.device.off()

    def alert_pattern(self):
        """SOS pattern or alert pattern"""
        if self.device:
            for _ in range(3):
                self.device.value = 1.0
                time.sleep(0.2)
                self.device.off()
                time.sleep(0.1)
