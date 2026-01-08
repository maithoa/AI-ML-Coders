import tensorflow as tf

#1. Print TensorFlow version and available devices
print("TensorFlow version:", tf.__version__)
print("Devices available:", tf.config.list_physical_devices())

# 2. Print the installation path of TensorFlow
print(f"Location: {tf.__file__}")

# 3. Check for GPU availability
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")