from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Suppose we have a list of opcodes for a file
opcodes = ['mov', 'add', 'mov', 'jmp']

# Reshape the list into a 2D array as expected by OneHotEncoder
opcodes_array = np.array(opcodes).reshape(-1, 1)

# Create and fit the encoder using the updated parameter name
encoder = OneHotEncoder(sparse_output=False)
one_hot = encoder.fit_transform(opcodes_array)

print("One-hot encoded matrix:")
print(one_hot)

# import tensorflow as tf

# # Set TensorFlow to use GPU, if available
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # Limit memory usage
#         )
#         print("Using GPU:", gpus[0])
#     except RuntimeError as e:
#         print(e)
# else:
#     print("No GPU detected, running on CPU.")


# import pickle
# from pathlib import Path

# filepath = Path("/home/016950414/cs298/saved_models/distilbert/mean_embedding_per_file.pkl")

# with filepath.open("rb") as f:
#     data = pickle.load(f)

# types = set(key[0] for key in data.keys())
# print(types, len(types))

# for key, value in data.items():
#     print(f"{key}: {value.shape}")