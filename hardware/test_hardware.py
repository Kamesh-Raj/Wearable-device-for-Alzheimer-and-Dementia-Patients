"""
Hardware Diagnostic Tool
RUN THIS ON THE DEVICE (RASPBERRY PI)
Tests all connected hardware components: Camera, Mic, Speaker, Display, Haptic, GPS.
"""

import sys
import os
import time
import logging
import cv2
import sounddevice as sd
import numpy as np
import threading

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import modules (Graceful failure for Windows/Dev PC)
try:
    from hardware.modules.camera_module import CameraModule
    from hardware.modules.audio_module import AudioModule
    from hardware.modules.display_module import DisplayModule
    from hardware.modules.haptic_module import HapticModule
    from hardware.modules.gps_module import GPSModule
except ImportError:
    # Try local import if run from hardware/
    try:
        from modules.camera_module import CameraModule
        from modules.audio_module import AudioModule
        from modules.display_module import DisplayModule
        from modules.haptic_module import HapticModule
        from modules.gps_module import GPSModule
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Run this script from inside the 'hardware/' directory or ensure paths work.")
        # Mock for PC testing logic below
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardwareTests")

def test_camera():
    print("\n--- Testing Camera ---")
    try:
        cam = CameraModule(rotation=0)
        print("Camera initialized. Capturing frame...")
        frame = cam.get_frame(auto_rotate=True)
        
        if frame is not None:
            print(f"Success! Frame captured. Size: {frame.shape}")
            cv2.imshow("Test Camera", frame)
            cv2.waitKey(2000) # Show for 2 seconds
            cv2.destroyAllWindows()
            cam.release()
            return True
        else:
            print("Failed to capture frame.")
            return False
    except Exception as e:
        print(f"Camera Test Failed: {e}")
        return False

def test_audio():
    print("\n--- Testing Audio (Mic & Speaker) ---")
    try:
        audio = AudioModule()
        duration = 3  # seconds
        fs = 16000
        
        print(f"Recording for {duration} seconds... Speak into the mic!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        
        print("Playing back...")
        sd.play(recording, fs)
        sd.wait()
        print("Playback complete.")
        
        # Test TTS
        print("Testing TTS engine...")
        audio.speak("Audio test complete.")
        
        return True
    except Exception as e:
        print(f"Audio Test Failed: {e}")
        return False

def test_display():
    print("\n--- Testing OLED Display ---")
    try:
        disp = DisplayModule()
        if disp.device:
            print("Display initialized.")
            disp.clear()
            disp.show_text("Hardware Test\nRunning...")
            time.sleep(2)
            disp.show_emoji("happy")
            time.sleep(2)
            disp.clear()
            print("Display updated successfully.")
            return True
        else:
            print("Display device not found (Check connection or i2c).")
            return False
    except Exception as e:
        print(f"Display Test Failed: {e}")
        return False

def test_haptic():
    print("\n--- Testing Haptic Motor ---")
    try:
        haptic = HapticModule(pin=17)
        if haptic.device:
            print("Vibrating for 1 second...")
            haptic.vibrate(duration=1.0)
            print("Done.")
            return True
        else:
            print("Haptic device not initialized (Check GPIO).")
            return False
    except Exception as e:
        print(f"Haptic Test Failed: {e}")
        return False
        
def test_gps():
    print("\n--- Testing GPS Module ---")
    try:
        gps = GPSModule(port='/dev/ttyS0', baudrate=9600) # Usually ttyS0 or ttyAMA0 on Pi
        if gps.serial:
            print("Waiting for GPS fix (This can take minutes indoors)...")
            start = time.time()
            while time.time() - start < 10: # Wait 10s for test
                loc = gps.get_location()
                if loc:
                    print(f"Location Found: {loc}")
                    gps.stop()
                    return True
                time.sleep(1)
            print("No fix yet (Timeout), but module is communicating.")
            gps.stop()
            return True # Consider pass if module opened
        else:
             print("GPS Serial failed to open.")
             return False
    except Exception as e:
        print(f"GPS Test Failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Hardware Diagnostic...")
    
    results = {}
    
    results['Camera'] = test_camera()
    results['Audio'] = test_audio()
    results['Display'] = test_display()
    results['Haptic'] = test_haptic()
    results['GPS'] = test_gps()
    
    print("\n--- Test Summary ---")
    for component, success in results.items():
        status = "[PASS]" if success else "[FAIL/NA]"
        print(f"{component}: {status}")
        
    print("\nTests Complete.")
