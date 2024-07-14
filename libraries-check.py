import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

print(hello.numpy())

print(hello.numpy().decode('utf-8'))

from tensorflow.keras import backend as Keras

hello = Keras.constant('Hello, Keras!', dtype=tf.string)

# Print the constant value
print(hello.numpy().decode('utf-8'))
