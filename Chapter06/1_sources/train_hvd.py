import os
import tensorflow as tf
import argparse
from mnist_setup import build_and_compile_cnn_model, mnist_dataset
import horovod.keras as hvd
from keras import backend as K


def _initiate_hvd():
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")


def _get_world_size():
    num_workers = len(os.getenv("SM_HOSTS"))
    num_gpu_devices = int(os.getenv("SM_NUM_GPUS", 0))
    return num_workers * num_gpu_devices


def main(args):

    _initiate_hvd()
    hvd_model = build_and_compile_cnn_model()

    global_batch_size = args.batch_size_per_device * _get_world_size()
    shareded_by_rank_dataset = mnist_dataset(global_batch_size, hvd)

    checkpoint_dir = os.getenv("SM_MODEL_DIR")
    # Put all the callbacks together.
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
    ]

    if hvd.rank() == 0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}"),
                save_weights_only=True,
            )
        )

    hvd_model.fit(
        shareded_by_rank_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch // hvd.size(),
        callbacks=callbacks,
    )

    # Save model only in one rank to avoid conflicts
    if hvd.rank() == 0:
        hvd_model.save(os.getenv("SM_MODEL_DIR"), save_format="tf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-per-device", type=int, default=2)
    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    parser.add_argument("--steps-per-epoch", type=int, default=70)
    args = parser.parse_args()

    main(args)
