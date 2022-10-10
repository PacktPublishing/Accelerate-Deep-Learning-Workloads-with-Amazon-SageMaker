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
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


@retry(
    stop_max_delay=1000 * 60 * 15,
    wait_exponential_multiplier=100,
    wait_exponential_max=30000,
)
def _dns_lookup(host):
    """Retry DNS lookup on host."""
    return socket.gethostbyname(host)


def _build_tf_config():

    hosts = json.loads(os.getenv("SM_HOSTS"))
    current_host = os.getenv("SM_CURRENT_HOST")

    workers = hosts

    def host_addresses(hosts, port=7777):
        return ["{}:{}".format(host, port) for host in hosts]

    tf_config = {"cluster": {}, "task": {}}
    tf_config["cluster"]["worker"] = host_addresses(workers)
    tf_config["task"] = {"index": workers.index(current_host), "type": "worker"}

    os.environ["TF_CONFIG"] = json.dumps(tf_config)


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

    for host in json.loads(os.getenv("SM_HOSTS")):
        _dns_lookup(host)
        print(f"DNS lookup complete for {host}")

    _set_nccl_environment()
    _build_tf_config()
    print(os.environ)

    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CollectiveCommunication.AUTO
        )
    )

    print(f"How many replicas in sync? {strategy.num_replicas_in_sync}")

    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

    global_batch_size = args.batch_size_per_device * _get_world_size()
    multi_worker_dataset = mnist_dataset(global_batch_size)

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

    multi_worker_model.fit(
        multi_worker_dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )

    # 2. Save model
    multi_worker_model.save(os.getenv("SM_MODEL_DIR"), save_format="tf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-per-device", type=int, default=2)
    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    parser.add_argument("--steps-per-epoch", type=int, default=70)
    args = parser.parse_args()

    main(args)
