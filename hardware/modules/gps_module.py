"""
GPS Module - Reads NMEA data from Serial GPS (e.g., NEO-6M)
"""

import serial
import time
import logging
import threading

class GPSModule:
    def __init__(self, port='/dev/ttyS0', baudrate=9600):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.current_location = None # (lat, lon)
        
        try:
            self.serial = serial.Serial(port, baudrate=baudrate, timeout=1)
            self.logger.info(f"GPS Serial opened on {port}")
            self.running = True
            
            # Start reader thread
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to open GPS serial: {e}")
            self.serial = None

    def _read_loop(self):
        import pynmea2 # Requires pip install pynmea2
        
        while self.running and self.serial:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore')
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        self.current_location = (msg.latitude, msg.longitude)
                        # self.logger.debug(f"GPS Update: {self.current_location}")
            except Exception as e:
                # self.logger.error(f"GPS Read Error: {e}")
                pass
            time.sleep(0.1)

    def get_location(self):
        """Return (lat, lon) or None"""
        return self.current_location

    def stop(self):
        self.running = False
        if self.serial:
            self.serial.close()
