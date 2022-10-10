from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import re

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import RMSprop
from sagemaker_tensorflow import PipeModeDataset

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 100
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
INPUT_TENSOR_NAME = (
    "inputs_input"  # needs to match the name of the first layer + "_input"
)


def keras_model_fn(
    learning_rate, weight_decay,
):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model is transformed into a TensorFlow Estimator before training and saved in a
    TensorFlow Serving SavedModel at the end of training.
    """
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            name="inputs",
            input_shape=(HEIGHT, WIDTH, DEPTH),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))

    opt = RMSprop(lr=learning_rate, decay=weight_decay)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, "train")


def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, "eval")


def _input(epochs, batch_size, channel, channel_name):
    """Uses the tf.data input pipeline for CIFAR-10 dataset."""
    mode = args.data_config[channel_name]["TrainingInputMode"]
    logging.info("Running {} in {} mode".format(channel_name, mode))
    dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")

    # Repeat infinitely.
    dataset = dataset.repeat()
    dataset = dataset.prefetch(10)

    # Parse records.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)

    # Potentially shuffle records.
    if channel_name == "train":
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()

    return {INPUT_TENSOR_NAME: image_batch}, label_batch


def _train_preprocess_fn(image):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    featdef = {
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(value, featdef)
    image = tf.decode_raw(example["image"], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]), tf.float32,
    )
    label = tf.cast(example["label"], tf.int32)
    image = _train_preprocess_fn(image)
    return image, tf.one_hot(label, NUM_CLASSES)


def save_model(model, output):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={"image": model.input}, outputs={"scores": model.output}
    )

    builder = tf.saved_model.builder.SavedModelBuilder(output + "/1/")
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        },
    )

    builder.save()
    logging.info("Model successfully saved at: {}".format(output))


def main(args):
    logging.info("getting data")
    train_dataset = train_input_fn()
    eval_dataset = eval_input_fn()
    # validation_dataset = validation_input_fn()

    logging.info("configuring model")
    model = keras_model_fn(
        args.learning_rate,
        args.weight_decay,
        # args.optimizer, args.momentum, mpi, hvd # TODO: remove this
    )

    callbacks = []
    callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    callbacks.append(ModelCheckpoint(args.output_dir + "/checkpoint-{epoch}.h5"))

    logging.info("Starting training")

    model.fit(
        x=train_dataset[0],
        y=train_dataset[1],
        steps_per_epoch=(num_examples_per_epoch("train") // args.batch_size),
        epochs=args.epochs,
        validation_data=eval_dataset,
        validation_steps=(num_examples_per_epoch("eval") // args.batch_size),
        callbacks=callbacks,
    )
    save_model(model, args.model_output_dir)


def num_examples_per_epoch(subset="train"):
    if subset == "train":
        return 40000
    elif subset == "eval":
        return 10000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--validation",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_EVAL"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The directory where the model will be stored.",
    )
    parser.add_argument(
        "--model_output_dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DIR")
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2e-4,
        help="Weight decay for convolutions.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of steps to use for training.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--data-config", type=json.loads, default=os.environ.get("SM_INPUT_DATA_CONFIG")
    )
    args = parser.parse_args()
    main(args)
