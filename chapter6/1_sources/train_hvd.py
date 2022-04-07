import multiprocessing
import os
import random
import tensorflow as tf
import json
import tensorflow_datasets as tfds
import argparse
from mnist_setup import build_and_compile_cnn_model, mnist_dataset
import socket
from retrying import retry
import horovod.keras as hvd
from keras import backend as K


@retry(
    stop_max_delay=1000 * 60 * 15,
    wait_exponential_multiplier=100,
    wait_exponential_max=30000,
)
def _dns_lookup(host):
    """Retry DNS lookup on host."""
    return socket.gethostbyname(host)


def _initiate_hvd():
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    return


def _set_nccl_environment():
    """Set NCCL environment variables for the container.
    https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs
    Args:
        network_interface_name: The name of the network interface to use for
            distributed training.
    """
    # Set the network interface for inter node communication
    os.environ["NCCL_SOCKET_IFNAME"] = os.environ["SM_NETWORK_INTERFACE_NAME"]
    # Disable IB transport and force to use IP sockets by default
    os.environ["NCCL_IB_DISABLE"] = "1"
    # Set to INFO for more NCCL debugging information
    os.environ["NCCL_DEBUG"] = "INFO"

    # To check env var passed to each rank
    os.environ["NCCL_DEBUG_SUBSYS"] = "ENV"

    # Just checking if this helps https://github.com/tensorflow/tensorflow/issues/34638
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    return


def _get_world_size():
    # assuming we are working on GPU devices only
    num_workers = len(os.getenv("SM_HOSTS"))
    num_gpu_devices = int(os.getenv("SM_NUM_GPUS", 0))
    return num_workers * num_gpu_devices


def main(args):

    # for host in json.loads(os.getenv("SM_HOSTS")):
    #    _dns_lookup(host)
    #    print(f"DNS lookup complete for {host}")

    # _set_nccl_environment()
    _initiate_hvd()
    print(os.environ)

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

    # 2. Save model
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
