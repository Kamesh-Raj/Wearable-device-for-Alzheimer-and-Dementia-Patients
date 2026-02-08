import tensorflow as tf
import numpy as np
import os

# --- INITIALIZATION ---
tf.compat.v1.disable_eager_execution()
v1 = tf.compat.v1

# --- 1. ARCHITECTURE: TinyMobileFaceNet ---
def bottleneck_block(x, out_c, kernel, stride, expansion, scope, training):
    with v1.variable_scope(scope):
        in_c = x.get_shape().as_list()[-1]
        # Expansion
        x = v1.keras.layers.Conv2D(in_c * expansion, 1, 1, use_bias=False, name='exp')(x)
        x = v1.keras.layers.BatchNormalization(name='exp_bn')(x, training=training)
        x = tf.nn.relu6(x)
        # Depthwise
        x = v1.keras.layers.DepthwiseConv2D(kernel, stride, padding='same', use_bias=False, name='dw')(x)
        x = v1.keras.layers.BatchNormalization(name='dw_bn')(x, training=training)
        x = tf.nn.relu6(x)
        # Project
        x = v1.keras.layers.Conv2D(out_c, 1, 1, use_bias=False, name='proj')(x)
        x = v1.keras.layers.BatchNormalization(name='proj_bn')(x, training=training)
        if stride == 1 and in_c == out_c:
            return x + x
        return x

def build_tiny_mobilefacenet(inputs, training=True):
    with v1.variable_scope('TinyMobileFaceNet', reuse=v1.AUTO_REUSE):
        x = v1.keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
        x = v1.keras.layers.BatchNormalization(name='bn1')(x, training=training)
        x = tf.nn.relu6(x)
        x = bottleneck_block(x, 32, 3, 1, 2, 'res1', training)
        x = bottleneck_block(x, 64, 3, 2, 4, 'res2', training)
        x = bottleneck_block(x, 128, 3, 2, 4, 'res3', training)
        x = v1.keras.layers.Conv2D(256, 1, use_bias=False, name='conv_final')(x)
        x = v1.keras.layers.BatchNormalization(name='bn_final')(x, training=training)
        x = tf.nn.relu6(x)
        x = v1.keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
        embeddings = v1.keras.layers.Dense(128, name='embeddings_output')(x)
        return embeddings

# --- 2. LOSS & ACCURACY ---
def arcface_loss(embedding, labels, out_num, s=64.0, m=0.5):
    with v1.variable_scope('arcface_loss', reuse=v1.AUTO_REUSE):
        weights = v1.get_variable('weights', [128, out_num], initializer=v1.keras.initializers.VarianceScaling())
        embedding_norm = tf.nn.l2_normalize(embedding, axis=1)
        weights_norm = tf.nn.l2_normalize(weights, axis=0)
        cos_t = tf.matmul(embedding_norm, weights_norm)
        sin_t = tf.sqrt(1.0 - tf.square(cos_t))
        cos_mt = s * (cos_t * np.cos(m) - sin_t * np.sin(m))
        mask = tf.one_hot(labels, depth=out_num)
        logits = tf.where(mask > 0, cos_mt, s * cos_t)
        return logits

def calculate_accuracy(logits, labels):
    # Top-1 Accuracy: Is the highest logit the correct class?
    predictions = tf.argmax(logits, axis=1)
    correct_prediction = tf.equal(predictions, labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --- 3. DATA INPUT ---
def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    parsed = tf.io.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed['image_raw'], channels=3)
    image = tf.image.resize(image, [112, 112])
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return image, parsed['label']

# --- 4. MAIN ---
def main():
    num_classes = 87542
    batch_size = 64
    
    dataset = tf.data.TFRecordDataset('train_subset.tfrecords').map(parse_function).shuffle(1000).batch(batch_size).repeat()
    iterator = v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()

    embeddings = build_tiny_mobilefacenet(image_batch, training=True)
    logits = arcface_loss(embeddings, label_batch, num_classes)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batch))
    accuracy = calculate_accuracy(logits, label_batch)
    
    optimizer = v1.train.AdamOptimizer(0.001)
    update_ops = v1.get_collection(v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    saver = v1.train.Saver()
    print("Starting Training...")
    with v1.Session() as sess:
        sess.run(v1.global_variables_initializer())
        for step in range(1, 10001):
            _, l_val, acc_val = sess.run([train_op, loss, accuracy])
            if step % 100 == 0:
                print(f"Step {step} | Loss: {l_val:.4f} | Accuracy: {acc_val*100:.2f}%")
            if step % 2000 == 0:
                saver.save(sess, './checkpoints/tiny_face', global_step=step)

        # --- 5. TFLITE CONVERSION (FIXED) ---
        print("Converting to INT8 TFLite...")
        # v1 bridge used to fix 'from_session' error
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [image_batch], [embeddings])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ([np.random.uniform(-1,1,(1,112,112,3)).astype(np.float32)] for _ in range(100))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        with open("caregiver_detector.tflite", "wb") as f:
            f.write(tflite_model)
        print("Model saved: caregiver_detector.tflite")

if __name__ == "__main__":
    main()