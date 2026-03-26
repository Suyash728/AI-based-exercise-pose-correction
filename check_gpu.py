import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())

gpus = tf.config.list_physical_devices('GPU')
print("GPUs Found:", gpus)

# Quick test — multiply large matrices on GPU
import time
with tf.device('/GPU:0'):
    a = tf.random.normal([5000, 5000])
    b = tf.random.normal([5000, 5000])
    start = time.time()
    c = tf.matmul(a, b)
    print(f"GPU matrix multiply took: {time.time()-start:.2f}s")