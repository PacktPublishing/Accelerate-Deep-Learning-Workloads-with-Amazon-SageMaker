import os
import tensorflow as tf
import numpy as np


def mnist_dataset(batch_size, hvd=None):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(60000)
        .repeat()
        .batch(batch_size)
    )

    # If hvd object is passed, shared data based on hvd.rank()
    if hvd:
        train_dataset = train_dataset.shard(hvd.size(), hvd.rank())

    return train_dataset


def build_and_compile_cnn_model(hvd=None):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    # If hvd object is passed, use HVD-enabled optimizer
    if hvd:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001 * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model
