import tensorflow as tf
import os

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    return model

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path,
#     save_weights_only=True,
#     verbose=1
# )
#
# model = create_model()
#
# model.fit(
#     train_images,
#     train_labels,
#     epochs=10,
#     validation_data=(test_images, test_labels),
#     callbacks=[cp_callback] # pass callback to training
# )

# # include the epoch in the file name. (uses `str.format`)
# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     # Save weights, every 5-epochs.
#     period=5)
#
# model = create_model()
# model.save_weights(checkpoint_path.format(epoch=5))
# model.fit(train_images, train_labels,
#           epochs = 50, callbacks = [cp_callback],
#           validation_data = (test_images,test_labels),
#           verbose=0)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = create_model()
# model.load_weights(latest)
# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
# saved_model_path = tf.contrib.saved_model.save_keras_model(model, ".\\saved_models")

# new_model = tf.keras.models.load_model('my_model.h5')
# new_model.summary()
#
# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

new_model = tf.contrib.saved_model.load_keras_model(".\\saved_models\\1560359906")
new_model.summary()

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
