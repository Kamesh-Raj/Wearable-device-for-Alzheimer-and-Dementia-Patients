import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from tensorflow.keras import layers, models

# --- 1. SETTINGS ---
BASE_PATH = r"C:\Alzheimer1\ai_models\training\data\WEDA Fall\WEDA-FALL-main\dataset"
HZ_FOLDERS = ["5Hz", "10Hz", "25Hz", "40Hz", "50Hz"] 
TIMESTAMPS_PATH = os.path.join(BASE_PATH, "fall_timestamps.csv")
TARGET_HZ = 50
TIME_STEPS = 125 
CHANNELS = 6    
NUM_CLASSES = 4 

def resample_data(data, original_hz, target_hz):
    if original_hz == target_hz: return data
    n_samples = len(data)
    target_n_samples = int(n_samples * (target_hz / original_hz))
    x = np.linspace(0, 1, n_samples)
    x_new = np.linspace(0, 1, target_n_samples)
    f = interp1d(x, data, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_new)

def load_universal_dataset():
    X, y = [], []
    ts_df = pd.read_csv(TIMESTAMPS_PATH)
    ts_df['filename'] = ts_df['filename'].str.replace('/', os.sep)

    for hz_folder in HZ_FOLDERS:
        original_hz = int(hz_folder.replace("Hz", ""))
        data_dir = os.path.join(BASE_PATH, hz_folder)
        if not os.path.exists(data_dir): continue
        
        print(f"📦 Processing {hz_folder}...")
        for activity in os.listdir(data_dir):
            activity_path = os.path.join(data_dir, activity)
            if not os.path.isdir(activity_path): continue
            
            label_map = {"D01": 1, "D02": 2}
            base_label = label_map.get(activity, 0)

            for file in os.listdir(activity_path):
                if not file.endswith(".csv"): continue
                df = pd.read_csv(os.path.join(activity_path, file))
                
                # --- FIX: Strictly select 6 channels or pad 3 to 6 ---
                raw_data = df.iloc[:, 1:].values 
                if raw_data.shape[1] == 3:
                    raw_data = np.hstack((raw_data, np.zeros((raw_data.shape[0], 3))))
                elif raw_data.shape[1] >= 6:
                    raw_data = raw_data[:, :6]
                else:
                    continue # Skip malformed files

                raw_data = resample_data(raw_data, original_hz, TARGET_HZ)
                raw_data = (raw_data - np.mean(raw_data, axis=0)) / (np.std(raw_data, axis=0) + 1e-7)

                rel_path = os.path.join(activity, file.split('_')[0])
                match = ts_df[ts_df['filename'].str.contains(rel_path, regex=False)]

                for i in range(0, len(raw_data), TIME_STEPS // 2):
                    window = raw_data[i : i + TIME_STEPS]
                    
                    # --- FIX: Ensure window length is exactly TIME_STEPS ---
                    if len(window) < TIME_STEPS:
                        padding = np.zeros((TIME_STEPS - len(window), CHANNELS))
                        window = np.vstack((window, padding))
                    elif len(window) > TIME_STEPS:
                        window = window[:TIME_STEPS]

                    # Verify final shape is (125, 6)
                    if window.shape == (TIME_STEPS, CHANNELS):
                        label = base_label
                        if activity.startswith("F") and not match.empty:
                            start_t, end_t = match.iloc[0]['start_time'], match.iloc[0]['end_time']
                            curr_t = (i + TIME_STEPS/2) / TARGET_HZ
                            if start_t <= curr_t <= end_t: label = 3
                        
                        X.append(window)
                        y.append(label)
                
    return np.array(X, dtype='float32'), np.array(y)

def build_and_train():
    X, y = load_universal_dataset()
    print(f"✅ Final Dataset Shape: {X.shape}") # Should be (N, 125, 6)

    model = models.Sequential([
        layers.Input(shape=(TIME_STEPS, CHANNELS)),
        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

    # Export INT8 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([X[i:i+1]] for i in range(100))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type, converter.inference_output_type = tf.int8, tf.int8
    
    with open("gait_universal_int8.tflite", "wb") as f:
        f.write(converter.convert())
    print("🚀 Success! Universal INT8 model saved.")

if __name__ == "__main__":
    build_and_train()