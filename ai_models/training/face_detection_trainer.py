import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- SETTINGS ---
MODEL_H5_PATH = "blazeface_wearable.h5"
TFLITE_OUTPUT_PATH = "blazeface_int8.tflite"
WIDER_PATH = r"C:\Alzheimer1\ai_models\training\data\Wider Face\WIDER_train\WIDER_train\images"
IMG_SIZE = 128

# 1. RE-DEFINE ARCHITECTURE (Must match your training script exactly)
def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0

def blaze_block(x, f, s=1):
    shortcut = x
    if s > 1:
        shortcut = layers.MaxPooling2D(s)(shortcut)
        shortcut = layers.Conv2D(f, 1)(shortcut)
    x = layers.DepthwiseConv2D(5, strides=s, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation(hard_swish)(layers.add([shortcut, x]))

def build_blazeface():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(24, 5, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(hard_swish)(x)
    x = blaze_block(x, 24)
    x = blaze_block(x, 24)
    x = blaze_block(x, 48, s=2)
    s16 = blaze_block(x, 48, s=2) 
    reg1 = layers.Conv2D(2 * 4, 3, padding='same')(s16)
    cls1 = layers.Conv2D(2 * 1, 3, padding='same')(s16)
    s8 = blaze_block(s16, 96, s=2)
    reg2 = layers.Conv2D(6 * 4, 3, padding='same')(s8)
    cls2 = layers.Conv2D(6 * 1, 3, padding='same')(s8)
    reg1 = layers.Reshape((16 * 16 * 2, 4))(reg1)
    reg2 = layers.Reshape((8 * 8 * 6, 4))(reg2)
    cls1 = layers.Reshape((16 * 16 * 2, 1))(cls1)
    cls2 = layers.Reshape((8 * 8 * 6, 1))(cls2)
    reg_out = layers.Concatenate(axis=1, name='reg')([reg1, reg2])
    cls_concat = layers.Concatenate(axis=1)([cls1, cls2])
    cls_out = layers.Activation('sigmoid', name='cls')(cls_concat)
    return models.Model(inputs, [reg_out, cls_out])

# 2. LOAD WEIGHTS ONLY
print(f"📂 Rebuilding architecture and loading weights from {MODEL_H5_PATH}...")
model = build_blazeface()
try:
    model.load_weights(MODEL_H5_PATH)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error: Could not load weights. {e}")
    exit()

# 3. REPRESENTATIVE DATASET GENERATOR
def representative_data_gen():
    count = 0
    for root, dirs, files in os.walk(WIDER_PATH):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')) and count < 100:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = (img.astype('float32') - 127.5) / 128.0
                yield [np.expand_dims(img, axis=0)]
                count += 1

# 4. CONVERT TO TFLITE (INT8)
print("🔄 Starting INT8 Quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"🚀 Success! Model exported as {TFLITE_OUTPUT_PATH}")
except Exception as e:
    print(f"❌ Conversion failed: {e}")