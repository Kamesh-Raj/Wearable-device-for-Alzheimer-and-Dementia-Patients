"""
Sensor Manager - Hardware abstraction layer for Raspberry Pi Zero 2W
Manages all sensor inputs and outputs in the stacked architecture
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any
import logging

try:
    import RPi.GPIO as GPIO
    import picamera
    import pyaudio
    import smbus
    import serial
    from luma.core.interface.serial import i2c
    from luma.oled.device import ssd1306
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("Hardware libraries not available - running in simulation mode")

class SensorManager:
    """Manages all hardware sensors and outputs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hardware_available = HARDWARE_AVAILABLE
        
        # Hardware components
        self.camera = None
        self.audio = None
        self.i2c_bus = None
        self.gps_serial = None
        self.oled = None
        
        # GPIO pins
        self.VIBRATION_PIN = 18
        self.SPEAKER_PIN = 19
        
    async def initialize(self):
        """Initialize all hardware components"""
        if not self.hardware_available:
            self.logger.info("Running in simulation mode")
            return
        
        try:
            # Initialize GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.VIBRATION_PIN, GPIO.OUT)
            GPIO.setup(self.SPEAKER_PIN, GPIO.OUT)
            
            # Initialize camera
            self.camera = picamera.PiCamera()
            self.camera.resolution = (640, 480)
            
            # Initialize I2C for MPU6050 and OLED
            self.i2c_bus = smbus.SMBus(1)
            
            # Initialize OLED display
            serial_interface = i2c(port=1, address=0x3C)
            self.oled = ssd1306(serial_interface)
            
            # Initialize GPS serial connection
            self.gps_serial = serial.Serial('/dev/ttyS0', 9600, timeout=1)
            
            # Initialize audio
            self.audio = pyaudio.PyAudio()
            
            self.logger.info("Hardware initialization complete")
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            self.hardware_available = False
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from Pi Camera"""
        if not self.hardware_available or not self.camera:
            # Return simulated frame
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        try:
            frame = np.empty((480, 640, 3), dtype=np.uint8)
            self.camera.capture(frame, 'rgb')
            return frame
        except Exception as e:
            self.logger.error(f"Camera capture failed: {e}")
            return None
    
    async def capture_audio(self, duration: float = 2.0) -> Optional[np.ndarray]:
        """Capture audio from INMP441 MEMS microphone"""
        if not self.hardware_available or not self.audio:
            # Return simulated audio
            return np.random.randn(int(16000 * duration))
        
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            
            frames = []
            for _ in range(int(16000 * duration / 1024)):
                data = stream.read(1024)
                frames.append(np.frombuffer(data, dtype=np.int16))
            
            stream.stop_stream()
            stream.close()
            
            return np.concatenate(frames)
            
        except Exception as e:
            self.logger.error(f"Audio capture failed: {e}")
            return None
    
    async def get_imu_data(self) -> Dict[str, np.ndarray]:
        """Get 6-axis data from MPU6050"""
        if not self.hardware_available or not self.i2c_bus:
            # Return simulated IMU data
            return {
                'accel': np.random.randn(3),
                'gyro': np.random.randn(3),
                'timestamp': asyncio.get_event_loop().time()
            }
        
        try:
            # MPU6050 register addresses
            ACCEL_XOUT_H = 0x3B
            GYRO_XOUT_H = 0x43
            
            # Read accelerometer data
            accel_raw = []
            for i in range(6):
                accel_raw.append(self.i2c_bus.read_byte_data(0x68, ACCEL_XOUT_H + i))
            
            # Read gyroscope data
            gyro_raw = []
            for i in range(6):
                gyro_raw.append(self.i2c_bus.read_byte_data(0x68, GYRO_XOUT_H + i))
            
            # Convert to proper values
            accel = np.array([
                (accel_raw[0] << 8 | accel_raw[1]) / 16384.0,
                (accel_raw[2] << 8 | accel_raw[3]) / 16384.0,
                (accel_raw[4] << 8 | accel_raw[5]) / 16384.0
            ])
            
            gyro = np.array([
                (gyro_raw[0] << 8 | gyro_raw[1]) / 131.0,
                (gyro_raw[2] << 8 | gyro_raw[3]) / 131.0,
                (gyro_raw[4] << 8 | gyro_raw[5]) / 131.0
            ])
            
            return {
                'accel': accel,
                'gyro': gyro,
                'timestamp': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            self.logger.error(f"IMU data read failed: {e}")
            return None
    
    async def get_gps_location(self) -> Optional[Dict[str, float]]:
        """Get GPS coordinates from NEO-6M"""
        if not self.hardware_available or not self.gps_serial:
            # Return simulated GPS data
            return {
                'latitude': 40.7128 + np.random.randn() * 0.001,
                'longitude': -74.0060 + np.random.randn() * 0.001,
                'timestamp': asyncio.get_event_loop().time()
            }
        
        try:
            line = self.gps_serial.readline().decode('ascii', errors='replace')
            if line.startswith('$GPGGA'):
                parts = line.split(',')
                if len(parts) > 6 and parts[2] and parts[4]:
                    lat = float(parts[2][:2]) + float(parts[2][2:]) / 60
                    lon = float(parts[4][:3]) + float(parts[4][3:]) / 60
                    
                    if parts[3] == 'S':
                        lat = -lat
                    if parts[5] == 'W':
                        lon = -lon
                    
                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'timestamp': asyncio.get_event_loop().time()
                    }
        except Exception as e:
            self.logger.error(f"GPS read failed: {e}")
        
        return None
    
    async def speak(self, message: str):
        """Output speech through PAM8403 amplifier and speaker"""
        if not self.hardware_available:
            self.logger.info(f"SPEAK: {message}")
            return
        
        try:
            # Use espeak for text-to-speech
            import subprocess
            subprocess.run(['espeak', message], check=True)
        except Exception as e:
            self.logger.error(f"Speech output failed: {e}")
    
    async def display_message(self, message: str):
        """Display message on OLED screen"""
        if not self.hardware_available or not self.oled:
            self.logger.info(f"DISPLAY: {message}")
            return
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            image = Image.new('1', (128, 64))
            draw = ImageDraw.Draw(image)
            
            # Simple text wrapping
            lines = message.split('\n')
            y = 0
            for line in lines:
                draw.text((0, y), line, fill=255)
                y += 12
            
            self.oled.display(image)
            
        except Exception as e:
            self.logger.error(f"OLED display failed: {e}")
    
    async def vibrate_alert(self):
        """Strong vibration for alerts"""
        await self._vibrate(duration=1.0, intensity=1.0)
    
    async def vibrate_gentle(self):
        """Gentle vibration for comfort"""
        await self._vibrate(duration=0.5, intensity=0.5)
    
    async def _vibrate(self, duration: float, intensity: float):
        """Control vibration motor"""
        if not self.hardware_available:
            self.logger.info(f"VIBRATE: {duration}s at {intensity} intensity")
            return
        
        try:
            # PWM control for vibration intensity
            pwm = GPIO.PWM(self.VIBRATION_PIN, 1000)
            pwm.start(intensity * 100)
            await asyncio.sleep(duration)
            pwm.stop()
        except Exception as e:
            self.logger.error(f"Vibration failed: {e}")
    
    def cleanup(self):
        """Clean up hardware resources"""
        if self.hardware_available:
            if self.camera:
                self.camera.close()
            if self.audio:
                self.audio.terminate()
            if self.gps_serial:
                self.gps_serial.close()
            GPIO.cleanup()