import tensorflow as tf
import os
import cv2
from tqdm import tqdm
import random

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_identity_limited_tfrecords(dataset_path, output_path, target_total_images=100000):
    writer = tf.io.TFRecordWriter(output_path)
    
    # Get list of all identity folders
    all_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    random.shuffle(all_folders) # Randomize which people we pick
    
    final_image_list = []
    current_count = 0
    selected_identities = 0

    print("Selecting identities to reach 100k images...")
    for folder in all_folders:
        if current_count >= target_total_images:
            break
            
        class_dir = os.path.join(dataset_path, folder)
        label = int(folder)
        
        # Get all images for this specific person
        images_in_folder = [os.path.join(class_dir, i) for i in os.listdir(class_dir) 
                            if i.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images_in_folder) > 0:
            for img_path in images_in_folder:
                final_image_list.append((img_path, label))
            
            current_count += len(images_in_folder)
            selected_identities += 1

    print(f"Selected {selected_identities} identities for a total of {len(final_image_list)} images.")
    random.shuffle(final_image_list) # Shuffle images so batches are mixed

    print("Writing TFRecords...")
    for img_path, label in tqdm(final_image_list):
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (112, 112))
        _, img_encoded = cv2.imencode('.jpg', img)
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_encoded.tobytes()),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    
    writer.close()
    print(f"File saved to {output_path}. Ready to train!")

if __name__ == "__main__":
    DATA_PATH = r'C:\Alzheimer1\ai_models\training\data\MS1MV2\ms1m-arcface'
    create_identity_limited_tfrecords(DATA_PATH, 'train_subset.tfrecords')