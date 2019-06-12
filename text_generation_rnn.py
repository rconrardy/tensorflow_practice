import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

print('Length of text: {} characters'.format(len(text)))

print(text[:250])
