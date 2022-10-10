import multiprocessing
import os
import random
import tensorflow as tf
import json
import tensorflow_datasets as tfds
import argparse


def _get_dataset(world_size, batch_size_per_device):
    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets["train"], datasets["test"]
    num_train_examples = info.splits["train"].num_examples
    num_test_examples = info.splits["test"].num_examples

    BUFFER_SIZE = 10000
    global_batch_size = batch_size_per_device * world_size

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = (
        mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(global_batch_size)
    )
    eval_dataset = mnist_test.map(scale).batch(global_batch_size)

    return train_dataset, eval_dataset


def _get_and_patch_tf_config():
    tf_config = json.loads(
        os.environ["TF_CONFIG"]
    )  # This has been prepared by SageMaker Training Toolkit

    tf_config["cluster"]["chief"] = tf_config["cluster"]["master"]
    tf_config["cluster"].pop("master")

    if tf_config["task"]["type"] == "master":
        tf_config["task"]["type"] = "chief"

    os.environ["TF_CONFIG"] = json.dumps(tf_config)

    return tf_config


def main(args):

    # 1. Run training
    tf_config = _get_and_patch_tf_config()
    print("TF Config after patching: \n")
    print(tf_config)

    # This is to enable variable sharding:
    variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=len(tf_config["cluster"]["ps"]),  # TODO: confirm that this works.
    )

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        tf.distribute.cluster_resolver.TFConfigClusterResolver(),
        variable_partitioner=variable_partitioner,
    )

    with strategy.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 3, activation="relu", input_shape=(28, 28, 1)
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

    def decay(epoch):
        if epoch < 3:
            return 1e-3
        elif epoch >= 3 and epoch < 7:
            return 1e-4
        else:
            return 1e-5

    checkpoint_dir = os.getenv("SM_MODEL_DIR")
    # Put all the callbacks together.
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(decay),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}"),
            save_weights_only=True,
        ),
    ]

    train_dataset, eval_dataset = _get_dataset(
        len(tf_config["cluster"]["worker"]), args.batch_size_per_device
    )

    model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)

    # 2. Run evaluation
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    eval_loss, eval_acc = model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval accuracy: {}".format(eval_loss, eval_acc))

    # 3. Save model
    model.save(os.getenv("SM_MODEL_DIR"), save_format="tf")

    # TODO: upload model artifacts to S3...


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-per-device", type=int, default=128)
    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    args = parser.parse_args()

    main(args)
