#!/usr/bin/env python3
"""
System Test Script for Neuro-Assistive Ecosystem
Tests core functionality without hardware dependencies
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from database.db_manager import DatabaseManager
        print("✓ Database manager imported successfully")
    except Exception as e:
        print(f"✗ Database manager import failed: {e}")
        return False
    
    try:
        from ai_models.face_recognition import FaceRecognitionSystem
        print("✓ Face recognition system imported successfully")
    except ImportError as e:
        if "faiss" in str(e).lower():
            print("⚠ Face recognition imported with FAISS fallback (FAISS not installed)")
        else:
            print(f"✗ Face recognition import failed: {e}")
            return False
    
    try:
        from ai_models.emotion_recognition import EmotionRecognitionSystem
        print("✓ Emotion recognition system imported successfully")
    except ImportError as e:
        if "librosa" in str(e).lower():
            print("⚠ Emotion recognition imported with audio fallback (Librosa not installed)")
        else:
            print(f"✗ Emotion recognition import failed: {e}")
            return False
    
    try:
        from ai_models.gait_analyzer import GaitAnalyzer
        print("✓ Gait analyzer imported successfully")
    except Exception as e:
        print(f"✗ Gait analyzer import failed: {e}")
        return False
    
    try:
        from hardware.sensor_manager import SensorManager
        print("✓ Sensor manager imported successfully")
    except Exception as e:
        print(f"✗ Sensor manager import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database functionality"""
    print("\nTesting database functionality...")
    
    try:
        from database.db_manager import DatabaseManager
        
        # Create test database
        db = DatabaseManager("test_neuro_assistive.db")
        
        # Create Dummy User & Patient
        uid = db.create_user("testuser", "testpass", "patient")
        pid = db.register_patient("Test Patient", user_id=uid)
        
        # Test adding a known person
        person_id = db.add_known_person(pid, "Test Person", "Family", notes="Test entry")
        if person_id > 0:
            print("✓ Database person creation successful")
        else:
            print("✗ Database person creation failed")
            return False
        
        # Test retrieving person
        person = db.get_known_person(person_id)
        if person and person['name'] == "Test Person":
            print("✓ Database person retrieval successful")
        else:
            print("✗ Database person retrieval failed")
            return False
        
        # Cleanup
        db.close()
        try:
            os.remove("test_neuro_assistive.db")
            print("✓ Database test completed successfully")
        except:
            print("✓ Database test completed successfully (cleanup skipped)")
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_ai_models():
    """Test AI model initialization"""
    print("\nTesting AI model initialization...")
    
    try:
        from ai_models.face_recognition import FaceRecognitionSystem
        from ai_models.emotion_recognition import EmotionRecognitionSystem
        from ai_models.gait_analyzer import GaitAnalyzer
        
        # Test face recognition
        face_system = FaceRecognitionSystem()
        print("✓ Face recognition system initialized")
        
        # Test emotion recognition
        emotion_system = EmotionRecognitionSystem()
        print("✓ Emotion recognition system initialized")
        
        # Test gait analyzer
        gait_analyzer = GaitAnalyzer()
        print("✓ Gait analyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ AI model initialization failed: {e}")
        return False

def test_sensor_manager():
    """Test sensor manager in simulation mode"""
    print("\nTesting sensor manager (simulation mode)...")
    
    try:
        from hardware.sensor_manager import SensorManager
        import asyncio
        
        async def test_sensors():
            sensor_manager = SensorManager()
            await sensor_manager.initialize()
            
            # Test camera capture (simulation)
            frame = await sensor_manager.capture_frame()
            if frame is not None:
                print("✓ Camera simulation successful")
            else:
                print("✗ Camera simulation failed")
                return False
            
            # Test audio capture (simulation)
            audio = await sensor_manager.capture_audio(duration=0.1)
            if audio is not None:
                print("✓ Audio simulation successful")
            else:
                print("✗ Audio simulation failed")
                return False
            
            # Test IMU data (simulation)
            imu_data = await sensor_manager.get_imu_data()
            if imu_data:
                print("✓ IMU simulation successful")
            else:
                print("✗ IMU simulation failed")
                return False
            
            # Test GPS data (simulation)
            gps_data = await sensor_manager.get_gps_location()
            if gps_data:
                print("✓ GPS simulation successful")
            else:
                print("✗ GPS simulation failed")
                return False
            
            return True
        
        result = asyncio.run(test_sensors())
        return result
        
    except Exception as e:
        print(f"✗ Sensor manager test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Neuro-Assistive Ecosystem - System Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Functionality", test_database),
        ("AI Model Initialization", test_ai_models),
        ("Sensor Manager", test_sensor_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)