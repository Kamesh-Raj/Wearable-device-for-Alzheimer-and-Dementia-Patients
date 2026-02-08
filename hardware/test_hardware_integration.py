"""
Hardware Integration Test
Tests the complete hardware system with assistive AI integration
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from hardware.sensor_manager import SensorManager
from hardware.ai_models.assistive_ai import AssistiveAI
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_sensor_manager():
    """Test sensor manager initialization and data capture"""
    logger.info("Testing Sensor Manager...")
    
    sensors = SensorManager()
    await sensors.initialize()
    
    # Test camera
    logger.info("Testing camera capture...")
    frame = await sensors.capture_frame()
    if frame is not None:
        logger.info(f"✓ Camera working - Frame shape: {frame.shape}")
    else:
        logger.error("✗ Camera failed")
    
    # Test audio
    logger.info("Testing audio capture...")
    audio = await sensors.capture_audio(duration=1.0)
    if audio is not None:
        logger.info(f"✓ Audio working - Audio shape: {audio.shape}")
    else:
        logger.error("✗ Audio failed")
    
    # Test IMU
    logger.info("Testing IMU...")
    imu_data = await sensors.get_imu_data()
    if imu_data is not None:
        logger.info(f"✓ IMU working - Accel: {imu_data['accel']}, Gyro: {imu_data['gyro']}")
    else:
        logger.error("✗ IMU failed")
    
    # Test GPS
    logger.info("Testing GPS...")
    gps_data = await sensors.get_gps_location()
    if gps_data is not None:
        logger.info(f"✓ GPS working - Lat: {gps_data['latitude']:.4f}, Lon: {gps_data['longitude']:.4f}")
    else:
        logger.error("✗ GPS failed")
    
    # Test outputs
    logger.info("Testing outputs...")
    await sensors.speak("Testing speech output")
    await sensors.display_message("Test\nMessage")
    await sensors.vibrate_gentle()
    logger.info("✓ Outputs tested")
    
    sensors.cleanup()
    logger.info("Sensor Manager test complete\n")


async def test_assistive_ai():
    """Test assistive AI system"""
    logger.info("Testing Assistive AI System...")
    
    # Initialize AI
    ai = AssistiveAI('config/assistive_ai_config.json')
    
    # Load models
    model_paths = {
        'face_recognition': 'models/face_recognition_95_int8.tflite',
        'emotion': 'models/face_emotion_95_int8.tflite',
        'voice_emotion': 'models/voice_emotion_optimized_int8.tflite',
        'gait': 'models/gait_analyzer_int8.tflite'
    }
    
    logger.info("Loading AI models...")
    if ai.load_models(model_paths):
        logger.info("✓ AI models loaded successfully")
    else:
        logger.warning("⚠ AI models not loaded (may be in simulation mode)")
    
    # Test with simulated data
    logger.info("Testing AI processing...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_audio = np.random.randn(16000)
    
    results = ai.process_frame(test_frame, test_audio)
    
    logger.info(f"✓ AI processing complete")
    logger.info(f"  - Timestamp: {results['timestamp']}")
    logger.info(f"  - Face detection: {results['face_detection']}")
    logger.info(f"  - Emotion: {results['emotion']}")
    logger.info(f"  - Recommendations: {len(results['recommendations'])} actions")
    
    # Test patient summary
    logger.info("Testing patient summary...")
    summary = ai.get_patient_summary()
    logger.info(f"✓ Patient summary generated")
    logger.info(f"  - Current emotion: {summary['current_state']['emotion']}")
    logger.info(f"  - Stress level: {summary['current_state']['stress_level']:.2f}")
    logger.info(f"  - Needs attention: {summary['current_state']['needs_attention']}")
    
    logger.info("Assistive AI test complete\n")


async def test_integrated_system():
    """Test complete integrated system"""
    logger.info("Testing Integrated System...")
    
    sensors = SensorManager()
    await sensors.initialize()
    
    ai = AssistiveAI('config/assistive_ai_config.json')
    
    model_paths = {
        'face_recognition': 'models/face_recognition_95_int8.tflite',
        'emotion': 'models/face_emotion_95_int8.tflite',
        'voice_emotion': 'models/voice_emotion_optimized_int8.tflite',
        'gait': 'models/gait_analyzer_int8.tflite'
    }
    
    ai.load_models(model_paths)
    
    logger.info("Running 10 processing cycles...")
    
    for i in range(10):
        # Capture data
        frame = await sensors.capture_frame()
        audio = await sensors.capture_audio(duration=0.5)
        
        # Process through AI
        results = ai.process_frame(frame, audio)
        
        # Handle recommendations
        for rec in results['recommendations']:
            if rec.priority >= 4:
                logger.info(f"HIGH PRIORITY: {rec.message}")
                await sensors.speak(rec.message)
        
        logger.info(f"Cycle {i+1}/10 - Emotion: {ai.patient_state.emotion}, "
                   f"Stress: {ai.patient_state.stress_level:.2f}")
        
        await asyncio.sleep(0.5)
    
    # Export report
    logger.info("Exporting test report...")
    ai.export_report('reports/integration_test_report.json', time_range=1)
    logger.info("✓ Report exported")
    
    sensors.cleanup()
    logger.info("Integrated system test complete\n")


async def test_performance():
    """Test system performance metrics"""
    logger.info("Testing Performance...")
    
    import time
    
    sensors = SensorManager()
    await sensors.initialize()
    
    ai = AssistiveAI('config/assistive_ai_config.json')
    
    model_paths = {
        'face_recognition': 'models/face_recognition_95_int8.tflite',
        'emotion': 'models/face_emotion_95_int8.tflite',
        'voice_emotion': 'models/voice_emotion_optimized_int8.tflite',
        'gait': 'models/gait_analyzer_int8.tflite'
    }
    
    ai.load_models(model_paths)
    
    # Measure processing time
    num_iterations = 50
    times = []
    
    logger.info(f"Running {num_iterations} iterations...")
    
    for i in range(num_iterations):
        frame = await sensors.capture_frame()
        
        start_time = time.time()
        results = ai.process_frame(frame, None)
        elapsed = time.time() - start_time
        
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{num_iterations}")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    logger.info("Performance Results:")
    logger.info(f"  - Average processing time: {avg_time*1000:.2f} ms")
    logger.info(f"  - Std deviation: {std_time*1000:.2f} ms")
    logger.info(f"  - Min time: {min_time*1000:.2f} ms")
    logger.info(f"  - Max time: {max_time*1000:.2f} ms")
    logger.info(f"  - Effective FPS: {fps:.2f}")
    
    # Check if meets Pi Zero 2W requirements
    if avg_time < 0.5:  # 2 FPS minimum
        logger.info("✓ Performance meets Pi Zero 2W requirements")
    else:
        logger.warning("⚠ Performance may be too slow for real-time operation")
    
    sensors.cleanup()
    logger.info("Performance test complete\n")


async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Hardware Integration Test Suite")
    logger.info("=" * 60 + "\n")
    
    try:
        # Run tests
        await test_sensor_manager()
        await test_assistive_ai()
        await test_integrated_system()
        await test_performance()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
