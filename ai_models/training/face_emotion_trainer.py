import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
CSV_PATH = r"C:\Alzheimer1\ai_models\training\data\CKplus\ckextended.csv"
IMAGE_DIR = r"C:\Alzheimer1\ai_models\training\data\CKplus"

# Unified Map: 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral
EMOTION_MAP = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5, 'neutral': 6}

def load_hybrid_data():
    X, y = [], []

    # A. Load from CSV
    if os.path.exists(CSV_PATH):
        print("Extracting pixels from CSV...")
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            pixels = np.asarray([int(p) for p in row['pixels'].split(' ')]).reshape(48, 48)
            X.append(pixels.astype('float32') / 255.0)
            y.append(row['emotion'])

    # B. Load from Folders
    print("Extracting images from local folders...")
    for folder_name, label in EMOTION_MAP.items():
        folder_path = os.path.join(IMAGE_DIR, folder_name)
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    # Using cv2-headless to read grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        X.append(img.astype('float32') / 255.0)
                        y.append(label)

    X = np.expand_dims(np.array(X), -1)
    # Filter out any labels outside 0-6 if necessary
    y = np.array(y)
    mask = y < 7
    X, y = X[mask], y[mask]
    
    y = tf.keras.utils.to_categorical(y, num_classes=7)
    return X, y

# --- 2. PI-OPTIMIZED ARCHITECTURE (MINI-XCEPTION) ---
def build_pi_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    
    # Entry Block
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual Depthwise Separable Blocks (Efficiency + Accuracy)
    for filters in [64, 128]:
        # Shortcut for Residual Connection
        residual = layers.Conv2D(filters, (1, 1), strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        # Depthwise Separable Path
        x = layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        
        x = layers.add([x, residual])

    # Global Average Pooling saves 90% parameters over Flatten
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 3. TRAINING AND INT8 EXPORT ---
if __name__ == "__main__":
    X, y = load_hybrid_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y.argmax(1))

    # Augmentation prevents overfitting (90/60 gap fix)
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, horizontal_flip=True)
    
    model = build_pi_model()
    print("Training 100 Epochs...")
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=100, validation_data=(X_val, y_val))

    # --- INT8 QUANTIZATION ---
    print("Quantizing for Pi Zero 2W...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def rep_data_gen():
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]
            
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save to models directory
    os.makedirs("models", exist_ok=True)
    model_path = "models/face_emotion_95_int8.tflite"
    with open(model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Export Complete: {model_path} is ready.")