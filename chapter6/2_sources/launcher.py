# This module gathers requirements parameters of pytorch distirbuted training world
# from environmental variable propagated by DSP for Pytorch Distributed job type.

# The module is intended to be light-weight and rely exclusively on native torch distributed utility:
# https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py


from argparse import ArgumentParser


import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
import logging
import json

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

# port for distributed processes to communicate
RDZV_PORT = "7777"


def parse_args():
    """
    TODO: likely to delete in future
    """
    parser = ArgumentParser(
        description="PyTorch DDP training launch "
        "helper utility that will spawn up "
        "multiple distributed processes"
    )
    parser.add_argument(
        "--train-script",
        type=str,
        help="Train script to run in distributed mode",
    )

    return parser.parse_known_args()


def main():
    distr_args, training_args = parse_args()

    LOGGER.info(f"Arguments: {distr_args}")

    # world size in terms of number of processes across all workers
    nodes = json.loads(os.getenv("SM_HOSTS", "[]"))
    nnodes = len(nodes)
    node_rank = nodes.index(os.getenv("SM_CURRENT_HOST"))
    nproc_per_node = os.getenv("SM_NUM_GPUS", 1)

    # Construction CMD to start torch distributed launcher
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--nnodes={str(nnodes)}",
        f"--node_rank={node_rank}",
        f"--rdzv_id={os.getenv('SAGEMAKER_JOB_NAME')}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={nodes[0]}:{RDZV_PORT}",
        distr_args.train_script,
    ]

    # Passing other CMD arguments to training script directly.
    cmd.extend(training_args)

    LOGGER.info(f"Training CMD to be executed on each node once:{cmd}")

    # Spawning DDP launcher process which will in it's turn create multiple child processes
    #
    process = subprocess.Popen(cmd, env=os.environ)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    main()
