import os
import cv2
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# --- 1. SETTINGS & PATHS ---
PATHS = {
    "voice": r"C:\Alzheimer1\ai_models\training\data\cremad",
    "face_emo": r"C:\Alzheimer1\ai_models\training\data\CKplus",
    "gait_base": r"C:\Alzheimer1\ai_models\training\data\WEDA Fall\WEDA-FALL-main\dataset",
    "face_det": r"C:\Alzheimer1\ai_models\training\data\Wider Face\WIDER_train\WIDER_train\images",
    "face_rec": r"C:\Alzheimer1\ai_models\training\data\MS1MV2\ms1m-arcface"
}

MODELS = {
    "detection": "blazeface_int8.tflite",
    "voice": "voice_emotion_optimized_int8.tflite",
    "face_emo": "face_emotion_95_int8.tflite",
    "gait": "gait_universal_int8.tflite",
    "recognition": "caregiver_detector.tflite"
}

# --- 2. CORE TFLITE INFERENCE ENGINE ---
def run_tflite_inference(model_path, X_test, multi_output=False):
    """Executes INT8 inference with manual quantization."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    in_scale, in_zero = input_details['quantization']
    
    results = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        # Pre-quantize input to INT8
        if input_details['dtype'] == np.int8:
            sample = np.round(sample / in_scale + in_zero).clip(-128, 127).astype(np.int8)
            
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        
        if multi_output:
            # Multi-output for BlazeFace [reg, cls]
            results.append([interpreter.get_tensor(od['index']) for od in output_details])
        else:
            # Standard classification or embedding output
            results.append(interpreter.get_tensor(output_details[0]['index']))
            
    return results

# --- 3. VALIDATION MODULES ---

def validate_face_detection():
    print("\n[1/5] Validating BlazeFace Detection...")
    folder = os.listdir(PATHS["face_det"])[0]
    img_path = os.path.join(PATHS["face_det"], folder, os.listdir(os.path.join(PATHS["face_det"], folder))[0])
    img_raw = cv2.imread(img_path)
    img_norm = (cv2.resize(img_raw, (128, 128)).astype('float32') - 127.5) / 128.0
    
    out = run_tflite_inference(MODELS["detection"], np.array([img_norm]), multi_output=True)[0]
    cls = out[1][0] # Confidence scores
    faces = np.sum(cls > 0.5)
    print(f"✅ Detection Status: Found {faces} face anchors with >0.5 confidence.")

def validate_voice():
    print("\n[2/5] Validating Voice Emotion...")
    MAP_CREMAD = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}
    X, y = [], []
    files = [f for f in os.listdir(PATHS["voice"]) if f.endswith(".wav")][:50]
    for f in files:
        parts = f.split('_')
        if len(parts) >= 3 and parts[2] in MAP_CREMAD:
            audio, _ = librosa.load(os.path.join(PATHS["voice"], f), sr=16000, duration=2.5)
            audio = librosa.util.normalize(audio)
            if len(audio) < 40000: audio = np.pad(audio, (0, 40000-len(audio)))
            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            delta = librosa.feature.delta(mfcc); delta2 = librosa.feature.delta(mfcc, order=2)
            X.append(np.concatenate((mfcc, delta, delta2), axis=0).T)
            y.append(MAP_CREMAD[parts[2]])
    if X:
        preds = [np.argmax(p) for p in run_tflite_inference(MODELS["voice"], np.array(X, dtype='float32'))]
        print(classification_report(y, preds, target_names=list(MAP_CREMAD.keys()), zero_division=0))

def validate_face_emotion():
    print("\n[3/5] Validating Face Emotion...")
    EMO_MAP = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}
    X, y = [], []
    for folder, label in EMO_MAP.items():
        p = os.path.join(PATHS["face_emo"], folder)
        if not os.path.exists(p): continue
        for img_name in os.listdir(p)[:20]:
            img = cv2.resize(cv2.imread(os.path.join(p, img_name), cv2.IMREAD_GRAYSCALE), (48, 48))
            X.append(np.expand_dims(img.astype('float32') / 255.0, -1))
            y.append(label)
    if X:
        preds = [np.argmax(p) for p in run_tflite_inference(MODELS["face_emo"], np.array(X))]
        print(classification_report(y, preds, target_names=list(EMO_MAP.keys()), zero_division=0))

def validate_gait():
    print("\n[4/5] Validating Gait Analysis...")
    X, y = [], []
    test_dir = os.path.join(PATHS["gait_base"], "50Hz", "D01") 
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir)[:3]:
            raw = pd.read_csv(os.path.join(test_dir, file)).iloc[:, 1:].values
            if raw.shape[1] == 3: raw = np.hstack((raw, np.zeros((raw.shape[0], 3))))
            elif raw.shape[1] >= 6: raw = raw[:, :6]
            raw = (raw - np.mean(raw, axis=0)) / (np.std(raw, axis=0) + 1e-7)
            # Strict window check to avoid inhomogeneous shape errors
            for i in range(0, len(raw) - 125, 125):
                window = raw[i:i+125]
                if window.shape == (125, 6):
                    X.append(window); y.append(1)
    if X:
        preds = [np.argmax(p) for p in run_tflite_inference(MODELS["gait"], np.array(X, dtype='float32'))]
        print(f"✅ Gait Accuracy: {accuracy_score(y, preds)*100:.2f}%")

def validate_face_recognition():
    print("\n[5/5] Validating Face Recognition...")
    all_folders = [f for f in os.listdir(PATHS["face_rec"]) if os.path.isdir(os.path.join(PATHS["face_rec"], f))]
    person_path = os.path.join(PATHS["face_rec"], all_folders[0])
    images = [os.path.join(person_path, i) for i in os.listdir(person_path) if i.endswith(('.jpg', '.png'))][:2]
    
    if len(images) < 2: return
    
    X = []
    for img_path in images:
        img = (cv2.resize(cv2.imread(img_path), (112, 112)).astype('float32') - 127.5) / 128.0
        X.append(img)
    
    # Get embeddings and de-quantize output for ArcFace similarity
    interpreter = tf.lite.Interpreter(model_path=MODELS["recognition"])
    interpreter.allocate_tensors()
    out_scale, out_zero = interpreter.get_output_details()[0]['quantization']
    
    raw_embs = run_tflite_inference(MODELS["recognition"], np.array(X))
    embs = [(p.astype(np.float32) - out_zero) * out_scale for p in raw_embs]
    embs = [e[0] / np.linalg.norm(e[0]) for e in embs] # Normalize to unit hypersphere
    
    dist = np.linalg.norm(embs[0] - embs[1])
    sim = np.dot(embs[0], embs[1])
    print(f"✅ Euclidean Dist: {dist:.4f} | Cosine Sim: {sim:.4f}")
    print("🚀 Verification Success" if dist < 1.1 else "⚠️ High Variance")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    validate_face_detection()
    validate_voice()
    validate_face_emotion()
    validate_gait()
    validate_face_recognition()