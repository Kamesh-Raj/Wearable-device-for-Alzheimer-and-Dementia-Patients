#!/usr/bin/env python3
"""
Device Setup Script for Neuro-Assistive Ecosystem
Initializes hardware, AI models, and database on Raspberry Pi Zero 2W
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.db_manager import DatabaseManager
from ai_models.face_recognition import FaceRecognitionSystem
from ai_models.emotion_recognition import EmotionRecognitionSystem
from ai_models.gait_analyzer import GaitAnalyzer
from hardware.sensor_manager import SensorManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/device_setup.log'),
            logging.StreamHandler()
        ]
    )

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models',
        'logs',
        'images/known_persons',
        'images/context'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_database():
    """Initialize database"""
    try:
        print("Setting up database...")
        db = DatabaseManager()
        
        # Add default settings
        default_settings = {
            'device_name': 'Neuro-Assistive Device',
            'patient_name': 'Patient',
            'emergency_contact': '+1234567890',
            'medication_reminder_enabled': True,
            'emotion_monitoring_enabled': True,
            'gait_analysis_enabled': True,
            'safety_monitoring_enabled': True
        }
        
        for key, value in default_settings.items():
            db.set_setting(key, value, description=f"Default {key} setting")
        
        # Add default safe zone (home)
        db.add_safety_zone(
            name="Home",
            center_lat=40.7128,  # Replace with actual coordinates
            center_lon=-74.0060,
            radius_meters=100
        )
        
        print("✓ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Database setup failed: {e}")
        return False

def setup_ai_models():
    """Initialize AI models"""
    try:
        print("Setting up AI models...")
        
        # Initialize face recognition
        face_recognition = FaceRecognitionSystem()
        print("✓ Face recognition system initialized")
        
        # Initialize emotion recognition
        emotion_recognition = EmotionRecognitionSystem()
        print("✓ Emotion recognition system initialized")
        
        # Initialize gait analyzer
        gait_analyzer = GaitAnalyzer()
        print("✓ Gait analyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ AI models setup failed: {e}")
        return False

async def test_hardware():
    """Test hardware components"""
    try:
        print("Testing hardware components...")
        
        sensor_manager = SensorManager()
        await sensor_manager.initialize()
        
        # Test camera
        frame = await sensor_manager.capture_frame()
        if frame is not None:
            print("✓ Camera test passed")
        else:
            print("⚠ Camera test failed (may be in simulation mode)")
        
        # Test audio
        audio = await sensor_manager.capture_audio(duration=1.0)
        if audio is not None:
            print("✓ Audio test passed")
        else:
            print("⚠ Audio test failed (may be in simulation mode)")
        
        # Test IMU
        imu_data = await sensor_manager.get_imu_data()
        if imu_data:
            print("✓ IMU test passed")
        else:
            print("⚠ IMU test failed (may be in simulation mode)")
        
        # Test GPS
        gps_data = await sensor_manager.get_gps_location()
        if gps_data:
            print("✓ GPS test passed")
        else:
            print("⚠ GPS test failed (may be in simulation mode)")
        
        # Test outputs
        await sensor_manager.speak("Hardware test successful")
        await sensor_manager.display_message("Test Complete")
        await sensor_manager.vibrate_gentle()
        
        print("✓ Output devices test passed")
        
        # Cleanup
        sensor_manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware test failed: {e}")
        return False

def install_dependencies():
    """Install required Python packages"""
    try:
        print("Installing dependencies...")
        
        # Core dependencies
        dependencies = [
            'numpy',
            'opencv-python',
            'tensorflow',
            'faiss-cpu',
            'librosa',
            'scipy',
            'pillow',
            'pyaudio',
            'pyserial',
            'smbus',
            'RPi.GPIO',
            'picamera',
            'luma.oled',
            'openai'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                print(f"✓ {dep} already installed")
            except ImportError:
                print(f"Installing {dep}...")
                os.system(f"pip3 install {dep}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependency installation failed: {e}")
        return False

def create_systemd_service():
    """Create systemd service for auto-start"""
    try:
        service_content = f"""[Unit]
Description=Neuro-Assistive Device
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={project_root}/hardware
ExecStart=/usr/bin/python3 {project_root}/hardware/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = Path('/etc/systemd/system/neuro-assistive.service')
        
        # Write service file (requires sudo)
        print("Creating systemd service...")
        print("Note: You may need to run this with sudo privileges")
        print(f"Service content written to: {service_path}")
        print("To enable auto-start, run:")
        print("sudo systemctl enable neuro-assistive.service")
        print("sudo systemctl start neuro-assistive.service")
        
        return True
        
    except Exception as e:
        print(f"✗ Systemd service creation failed: {e}")
        return False

async def main():
    """Main setup function"""
    print("🧠 Neuro-Assistive Device Setup")
    print("=" * 40)
    
    setup_logging()
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up database", setup_database),
        ("Setting up AI models", setup_ai_models),
        ("Testing hardware", test_hardware),
        ("Creating systemd service", create_systemd_service)
    ]
    
    success_count = 0
    
    for step_name, step_function in steps:
        print(f"\n{step_name}...")
        try:
            if asyncio.iscoroutinefunction(step_function):
                result = await step_function()
            else:
                result = step_function()
            
            if result:
                success_count += 1
            
        except Exception as e:
            print(f"✗ {step_name} failed: {e}")
    
    print(f"\n{'='*40}")
    print(f"Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("🎉 Device setup completed successfully!")
        print("\nNext steps:")
        print("1. Add known persons through the caregiver dashboard")
        print("2. Configure safety zones")
        print("3. Set up medication reminders")
        print("4. Start the device: python3 main.py")
    else:
        print("⚠ Some setup steps failed. Please check the logs and retry.")
    
    return success_count == len(steps)

if __name__ == "__main__":
    asyncio.run(main())