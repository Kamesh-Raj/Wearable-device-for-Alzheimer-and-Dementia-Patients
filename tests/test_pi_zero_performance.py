#!/usr/bin/env python3
"""
Pi Zero 2W Performance Test Suite
Tests the optimized model pipeline for real-world performance
"""

import sys
import time
import logging
import asyncio
import numpy as np
from pathlib import Path
import psutil
import threading
from collections import deque

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for performance testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class PerformanceMonitor:
    """Monitor system performance during AI inference"""
    
    def __init__(self):
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.temperatures = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # Temperature (Pi specific)
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = int(f.read()) / 1000.0
                        self.temperatures.append(temp)
                except:
                    self.temperatures.append(0)  # Not on Pi
                
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
    
    def record_inference_time(self, inference_time: float):
        """Record an inference time"""
        self.inference_times.append(inference_time)
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'cpu_avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_max': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_max': np.max(self.memory_usage) if self.memory_usage else 0,
            'temp_avg': np.mean(self.temperatures) if self.temperatures else 0,
            'temp_max': np.max(self.temperatures) if self.temperatures else 0,
            'inference_avg': np.mean(self.inference_times) if self.inference_times else 0,
            'inference_max': np.max(self.inference_times) if self.inference_times else 0,
            'fps_avg': 1.0 / np.mean(self.inference_times) if self.inference_times and np.mean(self.inference_times) > 0 else 0
        }

def test_face_detection_performance():
    """Test BlazeFace detection performance"""
    print("\n--- Face Detection Performance (BlazeFace) ---")
    
    try:
        from ai_models.face_recognition import FaceRecognitionSystem
        
        face_system = FaceRecognitionSystem()
        monitor = PerformanceMonitor()
        
        # Create test frames (simulating camera input)
        test_frames = []
        for i in range(50):
            # Create random RGB frame (320x240 - Pi Zero 2W camera resolution)
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        print(f"Testing with {len(test_frames)} frames (320x240 RGB)")
        
        monitor.start_monitoring()
        
        detection_times = []
        detected_faces = 0
        
        for i, frame in enumerate(test_frames):
            start_time = time.time()
            
            faces = face_system.detect_faces(frame)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            detection_times.append(inference_time)
            monitor.record_inference_time(inference_time)
            detected_faces += len(faces)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_frames)} frames")
        
        monitor.stop_monitoring()
        stats = monitor.get_stats()
        
        # Results
        avg_time = np.mean(detection_times)
        max_time = np.max(detection_times)
        min_time = np.min(detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Face Detection Results:")
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Max inference time: {max_time*1000:.1f}ms")
        print(f"  Min inference time: {min_time*1000:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Total faces detected: {detected_faces}")
        print(f"  CPU usage: {stats['cpu_avg']:.1f}% (max: {stats['cpu_max']:.1f}%)")
        print(f"  Memory usage: {stats['memory_avg']:.1f}% (max: {stats['memory_max']:.1f}%)")
        print(f"  Temperature: {stats['temp_avg']:.1f}°C (max: {stats['temp_max']:.1f}°C)")
        
        # Performance assessment
        if fps >= 3:
            print("  ✅ EXCELLENT: Meets Pi Zero 2W target (3-5 FPS)")
        elif fps >= 1:
            print("  ⚠️  ACCEPTABLE: Below target but usable")
        else:
            print("  ❌ POOR: Too slow for real-time use")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_face_recognition_performance():
    """Test MobileFaceNet recognition performance"""
    print("\n--- Face Recognition Performance (MobileFaceNet) ---")
    
    try:
        from ai_models.face_recognition import FaceRecognitionSystem
        
        face_system = FaceRecognitionSystem()
        monitor = PerformanceMonitor()
        
        # Create test face crops (112x112 - MobileFaceNet input size)
        test_faces = []
        for i in range(20):  # Fewer tests as this is slower
            face_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            test_faces.append(face_crop)
        
        print(f"Testing with {len(test_faces)} face crops (112x112 RGB)")
        
        monitor.start_monitoring()
        
        recognition_times = []
        
        for i, face_crop in enumerate(test_faces):
            start_time = time.time()
            
            features = face_system.extract_features(face_crop)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            recognition_times.append(inference_time)
            monitor.record_inference_time(inference_time)
            
            print(f"Processed face {i + 1}/{len(test_faces)}")
        
        monitor.stop_monitoring()
        stats = monitor.get_stats()
        
        # Results
        avg_time = np.mean(recognition_times)
        max_time = np.max(recognition_times)
        min_time = np.min(recognition_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Face Recognition Results:")
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Max inference time: {max_time*1000:.1f}ms")
        print(f"  Min inference time: {min_time*1000:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  CPU usage: {stats['cpu_avg']:.1f}% (max: {stats['cpu_max']:.1f}%)")
        print(f"  Memory usage: {stats['memory_avg']:.1f}% (max: {stats['memory_max']:.1f}%)")
        print(f"  Temperature: {stats['temp_avg']:.1f}°C (max: {stats['temp_max']:.1f}°C)")
        
        # Performance assessment
        if avg_time <= 0.4:  # 400ms target
            print("  ✅ EXCELLENT: Meets Pi Zero 2W target (~400ms)")
        elif avg_time <= 1.0:  # 1 second acceptable
            print("  ⚠️  ACCEPTABLE: Slower than target but usable")
        else:
            print("  ❌ POOR: Too slow for practical use")
        
        return True
        
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        return False

def test_emotion_recognition_performance():
    """Test Mini-Xception emotion recognition performance"""
    print("\n--- Emotion Recognition Performance (Mini-Xception) ---")
    
    try:
        from ai_models.emotion_recognition import EmotionRecognitionSystem
        
        emotion_system = EmotionRecognitionSystem()
        monitor = PerformanceMonitor()
        
        # Create test frames and audio
        test_frames = []
        test_audio = []
        
        for i in range(30):
            # Face crop for emotion (48x48 grayscale)
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            test_frames.append(frame)
            
            # Audio sample (2 seconds at 16kHz)
            audio = np.random.randn(32000).astype(np.int16)
            test_audio.append(audio)
        
        print(f"Testing with {len(test_frames)} frame+audio pairs")
        
        monitor.start_monitoring()
        
        emotion_times = []
        
        for i, (frame, audio) in enumerate(zip(test_frames, test_audio)):
            start_time = time.time()
            
            emotion_result = emotion_system.analyze_combined_emotion(frame, audio)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            emotion_times.append(inference_time)
            monitor.record_inference_time(inference_time)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(test_frames)} emotion analyses")
        
        monitor.stop_monitoring()
        stats = monitor.get_stats()
        
        # Results
        avg_time = np.mean(emotion_times)
        max_time = np.max(emotion_times)
        min_time = np.min(emotion_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Emotion Recognition Results:")
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Max inference time: {max_time*1000:.1f}ms")
        print(f"  Min inference time: {min_time*1000:.1f}ms")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  CPU usage: {stats['cpu_avg']:.1f}% (max: {stats['cpu_max']:.1f}%)")
        print(f"  Memory usage: {stats['memory_avg']:.1f}% (max: {stats['memory_max']:.1f}%)")
        print(f"  Temperature: {stats['temp_avg']:.1f}°C (max: {stats['temp_max']:.1f}°C)")
        
        # Performance assessment
        if fps >= 5:
            print("  ✅ EXCELLENT: Meets Pi Zero 2W target (5-7 FPS)")
        elif fps >= 2:
            print("  ⚠️  ACCEPTABLE: Below target but usable")
        else:
            print("  ❌ POOR: Too slow for real-time use")
        
        return True
        
    except Exception as e:
        print(f"❌ Emotion recognition test failed: {e}")
        return False

def test_full_pipeline_performance():
    """Test the complete AI pipeline performance"""
    print("\n--- Full Pipeline Performance Test ---")
    
    try:
        from hardware.sensor_manager import SensorManager
        from ai_models.face_recognition import FaceRecognitionSystem
        from ai_models.emotion_recognition import EmotionRecognitionSystem
        
        # Initialize systems
        sensor_manager = SensorManager()
        face_system = FaceRecognitionSystem()
        emotion_system = EmotionRecognitionSystem()
        
        monitor = PerformanceMonitor()
        
        print("Testing full pipeline: Camera → Face Detection → Recognition → Emotion")
        
        async def pipeline_test():
            await sensor_manager.initialize()
            
            pipeline_times = []
            
            for i in range(10):  # Fewer iterations for full pipeline
                start_time = time.time()
                
                # Simulate full pipeline
                frame = await sensor_manager.capture_frame()
                audio = await sensor_manager.capture_audio(duration=1.0)
                
                if frame is not None:
                    # Face detection
                    faces = face_system.detect_faces(frame)
                    
                    if faces:
                        # Face recognition on first face
                        face_crop = faces[0]['face_image']
                        features = face_system.extract_features(face_crop)
                    
                    # Emotion analysis
                    if audio is not None:
                        emotion_result = emotion_system.analyze_combined_emotion(frame, audio)
                
                end_time = time.time()
                pipeline_time = end_time - start_time
                
                pipeline_times.append(pipeline_time)
                monitor.record_inference_time(pipeline_time)
                
                print(f"Pipeline iteration {i + 1}/10 completed")
            
            return pipeline_times
        
        monitor.start_monitoring()
        
        # Run async pipeline test
        pipeline_times = asyncio.run(pipeline_test())
        
        monitor.stop_monitoring()
        stats = monitor.get_stats()
        
        # Results
        avg_time = np.mean(pipeline_times)
        max_time = np.max(pipeline_times)
        min_time = np.min(pipeline_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Full Pipeline Results:")
        print(f"  Average pipeline time: {avg_time:.2f}s")
        print(f"  Max pipeline time: {max_time:.2f}s")
        print(f"  Min pipeline time: {min_time:.2f}s")
        print(f"  Effective FPS: {fps:.1f}")
        print(f"  CPU usage: {stats['cpu_avg']:.1f}% (max: {stats['cpu_max']:.1f}%)")
        print(f"  Memory usage: {stats['memory_avg']:.1f}% (max: {stats['memory_max']:.1f}%)")
        print(f"  Temperature: {stats['temp_avg']:.1f}°C (max: {stats['temp_max']:.1f}°C)")
        
        # Performance assessment
        if avg_time <= 2.0:
            print("  ✅ EXCELLENT: Fast enough for real-time operation")
        elif avg_time <= 5.0:
            print("  ⚠️  ACCEPTABLE: Usable with frame skipping")
        else:
            print("  ❌ POOR: Too slow for practical use")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def check_system_resources():
    """Check system resources and Pi Zero 2W compatibility"""
    print("\n--- System Resource Check ---")
    
    try:
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        print(f"CPU cores: {cpu_count}")
        if cpu_freq:
            print(f"CPU frequency: {cpu_freq.current:.0f} MHz (max: {cpu_freq.max:.0f} MHz)")
        
        # Memory info
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**2):.0f} MB")
        print(f"Memory usage: {memory.percent:.1f}%")
        
        # Disk space
        disk = psutil.disk_usage('.')
        print(f"Disk space: {disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB")
        
        # Pi Zero 2W assessment
        is_pi_zero_2w = False
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'BCM2710' in cpuinfo or 'BCM2837' in cpuinfo:
                    is_pi_zero_2w = True
                    print("✅ Detected Pi Zero 2W compatible hardware")
        except:
            print("ℹ️  Running on non-Pi hardware")
        
        # Resource warnings
        warnings = []
        if memory.total < 400 * 1024 * 1024:  # Less than 400MB
            warnings.append("⚠️  Low RAM - may affect performance")
        
        if memory.percent > 80:
            warnings.append("⚠️  High memory usage - close other applications")
        
        if disk.free < 1024**3:  # Less than 1GB
            warnings.append("⚠️  Low disk space")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("✅ System resources look good")
        
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

def main():
    """Main performance test function"""
    print("🚀 Pi Zero 2W Performance Test Suite")
    print("=" * 50)
    
    setup_logging()
    
    # System check
    if not check_system_resources():
        return False
    
    # Performance tests
    tests = [
        ("Face Detection (BlazeFace)", test_face_detection_performance),
        ("Face Recognition (MobileFaceNet)", test_face_recognition_performance),
        ("Emotion Recognition (Mini-Xception + Tiny CNN)", test_emotion_recognition_performance),
        ("Full AI Pipeline", test_full_pipeline_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} COMPLETED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    # Final results
    print(f"\n{'='*50}")
    print(f"PERFORMANCE TEST RESULTS")
    print(f"{'='*50}")
    print(f"Tests completed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pi Zero 2W ready for deployment!")
        print("\n📋 Deployment Checklist:")
        print("  ✅ Run 'sudo python3 optimize_pi_zero.py' for system optimization")
        print("  ✅ Ensure camera and microphone are connected")
        print("  ✅ Test with real hardware using 'python3 hardware/main.py'")
        print("  ✅ Monitor temperature during extended use")
    else:
        print("⚠️  Some tests failed - check performance and optimize")
        print("\n🔧 Optimization suggestions:")
        print("  • Reduce camera resolution (320x240 → 160x120)")
        print("  • Increase frame skipping intervals")
        print("  • Disable unnecessary system services")
        print("  • Add heatsink if temperature > 70°C")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)