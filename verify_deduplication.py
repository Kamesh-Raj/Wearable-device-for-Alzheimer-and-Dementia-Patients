"""
Verification Script for Code Deduplication Changes
Tests all import paths and basic functionality after refactoring
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all critical imports"""
    print("=" * 60)
    print("Testing Import Paths")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Shared utilities
    try:
        from hardware.utils import ModelLoader, setup_logger, safe_execute
        print("[OK] hardware.utils imports successfully")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.utils import failed: {e}")
        tests.append(False)
    
    # Test 2: Gait Analyzer (consolidated)
    try:
        from hardware.ai_models.gait_analyzer import GaitAnalyzer
        analyzer = GaitAnalyzer()
        # Check intervention methods exist
        assert hasattr(analyzer, 'should_intervene'), "Missing should_intervene method"
        assert hasattr(analyzer, 'get_intervention_type'), "Missing get_intervention_type method"
        assert hasattr(analyzer, 'execute_intervention'), "Missing execute_intervention method"
        print("[OK] hardware.ai_models.gait_analyzer imports and has intervention methods")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.ai_models.gait_analyzer import/test failed: {e}")
        tests.append(False)
    
    # Test 3: Face Recognition System
    try:
        from hardware.ai_models.face_recognition import FaceRecognitionSystem
        print("[OK] hardware.ai_models.face_recognition imports successfully")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.ai_models.face_recognition import failed: {e}")
        tests.append(False)
    
    # Test 4: Emotion Recognition
    try:
        from hardware.ai_models.emotion_recognition import EmotionRecognitionSystem
        print("[OK] hardware.ai_models.emotion_recognition imports successfully")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.ai_models.emotion_recognition import failed: {e}")
        tests.append(False)
    
    # Test 5: Assistive AI
    try:
        from hardware.ai_models.assistive_ai import AssistiveAI
        print("[OK] hardware.ai_models.assistive_ai imports successfully")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.ai_models.assistive_ai import failed: {e}")
        tests.append(False)
    
    # Test 6: Caregiver Recognition (fixed imports)
    try:
        from database.db_manager import DatabaseManager
        from hardware.modules.caregiver_recognition import CaregiverRecognition
        print("[OK] hardware.modules.caregiver_recognition imports successfully")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.modules.caregiver_recognition import failed: {e}")
        tests.append(False)
    
    # Test 7: Emotion AI Module
    try:
        from hardware.modules.emotion_ai import EmotionAI
        print("[OK] hardware.modules.emotion_ai imports successfully")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] hardware.modules.emotion_ai import failed: {e}")
        tests.append(False)
    
    # Test 8: Database Manager (with helpers)
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        # Check helper methods exist
        assert hasattr(db, '_execute_query'), "Missing _execute_query helper"
        assert hasattr(db, '_serialize_json'), "Missing _serialize_json helper"
        assert hasattr(db, '_deserialize_json'), "Missing _deserialize_json helper"
        print("[OK] database.db_manager imports and has helper methods")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] database.db_manager import/test failed: {e}")
        tests.append(False)
    
    print("\n" + "=" * 60)
    print(f"Import Tests: {sum(tests)}/{len(tests)} passed")
    print("=" * 60)
    
    return all(tests)

def test_file_structure():
    """Verify file structure changes"""
    print("\n" + "=" * 60)
    print("Testing File Structure")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Duplicate gait_analyzer.py removed
    duplicate_gait = Path("hardware/modules/gait_analyzer.py")
    if not duplicate_gait.exists():
        print("[OK] Duplicate hardware/modules/gait_analyzer.py removed")
        tests.append(True)
    else:
        print("[FAIL] Duplicate hardware/modules/gait_analyzer.py still exists")
        tests.append(False)
    
    # Test 2: Shared utilities created
    utils_file = Path("hardware/utils.py")
    if utils_file.exists():
        print("[OK] hardware/utils.py created")
        tests.append(True)
    else:
        print("[FAIL] hardware/utils.py not found")
        tests.append(False)
    
    # Test 3: AI models in correct location
    ai_models_dir = Path("hardware/ai_models")
    required_files = [
        "gait_analyzer.py",
        "face_recognition.py",
        "emotion_recognition.py",
        "assistive_ai.py",
        "multimodal_fusion.py",
        "personal_fine_tuning.py"
    ]
    
    all_exist = all((ai_models_dir / f).exists() for f in required_files)
    if all_exist:
        print(f"[OK] All {len(required_files)} AI model files in hardware/ai_models/")
        tests.append(True)
    else:
        missing = [f for f in required_files if not (ai_models_dir / f).exists()]
        print(f"[FAIL] Missing AI model files: {missing}")
        tests.append(False)
    
    print("\n" + "=" * 60)
    print(f"File Structure Tests: {sum(tests)}/{len(tests)} passed")
    print("=" * 60)
    
    return all(tests)

def test_functionality():
    """Test basic functionality"""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    tests = []
    
    # Test 1: ModelLoader functionality
    try:
        from hardware.utils import ModelLoader
        loader = ModelLoader()
        # Should not crash
        result = loader.find_model("nonexistent_model.tflite")
        assert result is None, "Should return None for nonexistent model"
        print("[OK] ModelLoader.find_model() works correctly")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] ModelLoader test failed: {e}")
        tests.append(False)
    
    # Test 2: Logger setup
    try:
        from hardware.utils import setup_logger
        logger = setup_logger("test_logger")
        logger.info("Test message")
        print("[OK] setup_logger() works correctly")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] setup_logger test failed: {e}")
        tests.append(False)
    
    # Test 3: GaitAnalyzer intervention threshold
    try:
        from hardware.ai_models.gait_analyzer import GaitAnalyzer
        analyzer = GaitAnalyzer()
        assert analyzer.intervention_threshold == 0.8, "Intervention threshold not set"
        assert analyzer.intervention_cooldown_minutes == 30, "Cooldown not set"
        print("[OK] GaitAnalyzer intervention attributes initialized")
        tests.append(True)
    except Exception as e:
        print(f"[FAIL] GaitAnalyzer intervention test failed: {e}")
        tests.append(False)
    
    print("\n" + "=" * 60)
    print(f"Functionality Tests: {sum(tests)}/{len(tests)} passed")
    print("=" * 60)
    
    return all(tests)

def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("CODE DEDUPLICATION VERIFICATION")
    print("=" * 60 + "\n")
    
    import_success = test_imports()
    structure_success = test_file_structure()
    functionality_success = test_functionality()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Import Tests: {'PASS' if import_success else 'FAIL'}")
    print(f"File Structure Tests: {'PASS' if structure_success else 'FAIL'}")
    print(f"Functionality Tests: {'PASS' if functionality_success else 'FAIL'}")
    print("=" * 60)
    
    if import_success and structure_success and functionality_success:
        print("\nALL TESTS PASSED! Code deduplication successful.")
        return 0
    else:
        print("\nSOME TESTS FAILED. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
