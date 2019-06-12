import tensorflow as tf
import numpy as np

imdb = tf.keras.datasets.imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(x_train), len(y_train)))
print("Testing entries: {}, labels: {}".format(len(x_test), len(y_test)))

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

x_train = tf.keras.preprocessing.sequence.pad_sequence(
    x_train,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

x_test = tf.keras.preprocessing.sequence.pad_sequence(
    x_test,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

vocab_size = 10000

model = tf.keras.Sequential(
    tf.keras.layers.Embedding(vocab_size, 16)
    tf.keras.layers.GlobalAveragePooling1D()
    
)
