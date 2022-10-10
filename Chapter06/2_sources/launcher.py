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

# port for distributed DDP processes to communicate
RDZV_PORT = "7777"


def parse_args():
    parser = ArgumentParser(
        description="Custom arg parser. Using it to get reference to train script."
    )
    parser.add_argument(
        "--train-script",
        type=str,
        help="Train script to run in distributed mode",
    )

    return parser.parse_known_args()


def main():
    distr_args, training_hyperparameters = parse_args()

    # world size in terms of number of processes across all workers
    nodes = json.loads(os.getenv("SM_HOSTS"))
    nnodes = len(nodes)
    node_rank = nodes.index(os.getenv("SM_CURRENT_HOST"))
    nproc_per_node = os.getenv("SM_NUM_GPUS", 1)

    # Construct command line to to launch training processes using torch.distributed.run
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

    # Adding training hyperparameters which will be then passed in training script
    cmd.extend(training_hyperparameters)

    LOGGER.info(f"Command line to be executed on each node once:{cmd}")

    # Spawning DDP launcher process which will then spawn multiple child processes (one process per GPU)
    process = subprocess.Popen(cmd, env=os.environ)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == "__main__":
    main()
