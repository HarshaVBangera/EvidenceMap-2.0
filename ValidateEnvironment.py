import os

# Simple utility to validate the environment is capable of executing TensorFlow or PyTorch with GPUs.

# Set TensorFlow log level to suppress detailed logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TensorFlow checks
try:
    import tensorflow as tf

    print("TensorFlow is installed.")
    print("TensorFlow version:", tf.__version__)
    print("The following devices have been detected by TensorFlow and will be used:")
    if tf.__version__.startswith('1.'):
        from tensorflow.python.client import device_lib

        tf_devices = device_lib.list_local_devices()

        for device in tf_devices:
            print(f"{device.device_type}: {device.name}")
    elif tf.__version__.startswith('2.'):
        tf_devices = tf.config.list_physical_devices()
        for device in tf_devices:
            print(f"{device[1]}: {device.name}")

except ImportError:
    print("TensorFlow is not installed")

# PyTorch checks
try:
    import torch

    print("\nPyTorch is installed.")
    print("PyTorch version:", torch.__version__)
    print("\nListing available devices for PyTorch...")
    num_cpus = torch.get_num_threads()
    print(f"Number of CPU threads for PyTorch: {num_cpus}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs for PyTorch: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i} for PyTorch: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs found for PyTorch.")

except ImportError:
    print("\nPyTorch is not installed.")
