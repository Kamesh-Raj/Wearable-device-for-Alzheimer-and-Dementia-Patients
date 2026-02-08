import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION & PATHS ---
RAVDESS_DIR = r"C:\Alzheimer1\ai_models\training\data\ravdess"
CREMAD_DIR = r"C:\Alzheimer1\ai_models\training\data\cremad"
SR = 16000
DURATION = 2.5
TOTAL_SAMPLES = int(SR * DURATION)

# 0: Neutral, 1: Happy, 2: Sad, 3: Angry, 4: Fear, 5: Disgust, 6: Surprise
MAP_RAVDESS = {"01": 0, "02": 0, "03": 1, "04": 2, "05": 3, "06": 4, "07": 5, "08": 6}
MAP_CREMAD = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}

# --- 2. REFINED FEATURE EXTRACTION ---
def extract_120_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
        
        # Ensure consistent length (pad if shorter than 2.5s)
        if len(audio) < TOTAL_SAMPLES:
            audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)), 'constant')
        
        audio = librosa.util.normalize(audio)
        
        # 40 MFCCs + Velocity + Acceleration = 120 Features
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate and Transpose for 1D CNN Input: (TimeSteps, 120)
        features = np.concatenate((mfcc, delta, delta2), axis=0).T 
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 3. DATA LOADING ---
def load_data():
    X, y = [], []
    print("🚀 Loading Speech Datasets...")
    
    # RAVDESS
    for root, _, files in os.walk(RAVDESS_DIR):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split('-')
                if len(parts) >= 3:
                    code = parts[2]
                    if code in MAP_RAVDESS:
                        feat = extract_120_features(os.path.join(root, file))
                        if feat is not None:
                            X.append(feat)
                            y.append(MAP_RAVDESS[code])
    
    # CREMA-D
    for file in os.listdir(CREMAD_DIR):
        if file.endswith(".wav") and "_" in file:
            code = file.split('_')[2]
            if code in MAP_CREMAD:
                feat = extract_120_features(os.path.join(CREMAD_DIR, file))
                if feat is not None:
                    X.append(feat)
                    y.append(MAP_CREMAD[code])
                    
    return np.array(X), np.array(y)

# --- 4. LIGHTWEIGHT RESIDUAL ARCHITECTURE ---
def build_residual_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Block 1 - Initial Feature Extraction
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual Block with higher capacity
    res = layers.Conv1D(128, 1, padding='same')(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.Add()([x, res])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Global Pooling & Strong Regularization
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x) # Increased L2
    x = layers.Dropout(0.5)(x) # Increased Dropout
    outputs = layers.Dense(7, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Lower LR for stability
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y)
    
    model = build_residual_model(X_train.shape[1:])
    
    # Training with Early Stopping
    stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_data=(X_val, y_val), callbacks=[stop])

    # --- 5. ENHANCED INT8 EXPORT ---
    print("🔄 Exporting INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Diverse calibration data
    def representative_dataset_gen():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]
            
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    with open("voice_emotion_optimized_int8.tflite", "wb") as f:
        f.write(converter.convert())
    print("✅ Model saved as voice_emotion_optimized_int8.tflite")