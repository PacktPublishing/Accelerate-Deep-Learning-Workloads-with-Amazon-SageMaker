# This code is based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import division, print_function

import argparse
import ast
import os
import random
import logging

import numpy as np
import smdistributed.modelparallel.torch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR
from torchnet.dataset import SplitDataset
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 2  # two classes: ants and bees

## Make cudnn deterministic in order to get the same losses across runs.
## The following two lines can be removed if they cause a performance impact.
## For more details, see:
## https://pytorch.org/docs/stable/notes/randomness.html#cudnn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# SM Distributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, data, target):
    # TODO: not using autocase or any AMP
    # with autocast(1 > 0):
    #    output = model(data)
    output = model(data)
    # original loss from SM primer
    #     loss = F.nll_loss(output, target, reduction="mean")
    loss = F.cross_entropy(output, target, reduction="mean")

    # scaled_loss = loss  # TODO: it's not scaling it... need to do something about it
    model.backward(
        loss
    )  #  instead of usual loss.backward(), so SMP can control loss calculations
    return output, loss


def train(model, device, train_loader, optimizer, epoch, global_rank):
    model.train()  # set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        # SM Distributed: Move input tensors to the GPU ID used by the current process,
        # based on the set_device call.
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Return value, loss_mb is a loss for every minibatch
        _, loss_mb = train_step(model, data, target)

        # SM Distributed: Average the loss across microbatches.
        loss = loss_mb.reduce_mean()

        optimizer.step()

        if global_rank == 0 and batch_idx % 10 == 0:
            LOGGER.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


# SM Distributed: Define smp.step for evaluation.
@smp.step
def test_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
    pred = output.argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # SM Distributed: Moves input tensors to the GPU ID used by the current process
            # based on the set_device call.
            data, target = data.to(device), target.to(device)

            # Since test_step returns scalars instead of tensors,
            # test_step decorated with smp.step will return lists instead of StepOutput objects.
            loss_batch, correct_batch = test_step(model, data, target)
            test_loss += sum(loss_batch)
            correct += sum(correct_batch)

    test_loss /= len(test_loader.dataset)
    LOGGER.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def _cast_dict_from_string(input_dict: dict) -> argparse.Namespace:
    """
    this method attempts to convert dictionary of string values
    Namespace object with boolean and integer values when possible.

    Input: {'arg_bool': 'False', 'arg_int': '4', 'arg_str':'string'}
    Namespace(arg_bool=False, arg_int=4, arg_str='string')
    """

    output_args = argparse.Namespace()

    for key, value in input_dict.items():
        try:
            new_value = ast.literal_eval(value)
        except:
            new_value = value
        setattr(output_args, key, new_value)
    return output_args


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--mp_parameters", type=str, default="", help="Dictionary with SDT parameters"
    )
    return parser.parse_known_args()


def get_dataloaders(args, sdmp_args):
    # Data augmentation and normalization for training
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.environ[f"SM_CHANNEL_{x.upper()}"], data_transforms[x]
        )
        for x in ["train", "val"]
    }

    if sdmp_args.dp_size > 1:
        partitions_dict = {
            f"{i}": 1 / sdmp_args.dp_size for i in range(sdmp_args.dp_size)
        }
        dataset_train = SplitDataset(
            image_datasets["train"], partitions=partitions_dict
        )
        dataset_train.select(f"{sdmp_args.dp_rank}")

    elif (not sdmp_args.ddp) or sdmp_args.dp_size == 1:
        dataset_train = image_datasets["train"]
    else:
        raise Exception("Check your distributed configuration.")

    ##    {"num_workers": 1, "pin_memory": True, "shuffle": False, "drop_last": True}
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = None
    # Test loader will be available for global rank 0 process
    if sdmp_args.rank == 0:
        test_loader = torch.utils.data.DataLoader(
            image_datasets["val"],
            batch_size=args.test_batch_size,
            shuffle=True,
            drop_last=True,
        )
    return train_loader, test_loader


def get_resnet_model():
    # TODO: change model to resnet.
    # getting error like below on resnet now:
    #  on line with relu operation
    #  result = torch.relu_(input)
    #  ErrorMessage ":RuntimeError: Output 0 of SMPInputBackward is a view and is being modified inplace.
    #  This view was created inside a custom Function (or because an input was returned as-is) and the autograd
    #  logic to handle view+inplace would override the custom backward associated with the custom Function,
    #  leading to incorrect gradients. This behavior is forbidden. You can fix this by cloning the output of the custom Function.
    # model_ft = models.resnet18(pretrained=True)
    model_ft = models.squeezenet1_0(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    input_size = 224  # defined by Resnet18

    return model_ft, input_size


def get_squeezenet_model():
    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.classifier[1] = nn.Conv2d(
        512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1)
    )
    model_ft.num_classes = NUM_CLASSES
    input_size = 224

    return model_ft, input_size


def main():

    smp.init()

    args, unknown_args = parse_args()
    # this is parsing MP parameters passed as string by SageMaker
    # E.g. "--mp_parameters","auto_partition=True,ddp=False,microbatches=8,optimize=speed,partitions=2,pipeline=interleaved,placement_strategy=cluster"
    sdmp_args_string = dict(item.split("=") for item in args.mp_parameters.split(","))
    sdmp_args = _cast_dict_from_string(sdmp_args_string)
    sdmp_args.dp_size = smp.dp_size()
    sdmp_args.dp_rank = smp.dp_rank()
    sdmp_args.rank = smp.rank()

    LOGGER.info(
        f"Collected hyperparameters: {args}. \n Collected Model Parallel parameters: {sdmp_args}.\n"
        f"Following args are not parsed correctly and won't be used: {unknown_args}."
    )

    if not torch.cuda.is_available():
        raise ValueError("The script requires CUDA support, but CUDA not available")

    LOGGER.info(
        f"Hello from global rank {sdmp_args.rank}.",
        f"local rank {smp.local_rank()} and local size {smp.local_size()}",
        f"List of ranks where current model is stored {smp.get_mp_group()}",
        f"list of ranks with different replicas of the same model {smp.get_dp_group()}",
        f"current MP rank {smp.mp_rank()} and MP size is {smp.mp_size()}",
        f"current DP rank {smp.dp_rank()} and DP size is {smp.dp_size()}",
        f"Other params: {smp.tp_rank()}, {smp.tp_size()}, {smp.pp_rank()}, {smp.pp_size()}, {smp.rdp_rank()}, {smp.rdp_size()}",
    )

    # TODO: change it
    # model, args.input_size = get_resnet_model()
    model, args.input_size = get_squeezenet_model()

    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")

    optimizer = optim.Adadelta(model.parameters(), lr=4.0)
    model = smp.DistributedModel(model)
    optimizer = smp.DistributedOptimizer(optimizer)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # Scaling batch size if data parallelism used
    if sdmp_args.dp_size > 1:
        old_batch_size = args.batch_size
        args.batch_size //= sdmp_args.dp_size
        args.batch_size = max(args.batch_size, 1)
        LOGGER.info(f"Scaled batch size from {old_batch_size} to {args.batch_size}.")
    train_loader, test_loader = get_dataloaders(args, sdmp_args)

    for epoch in range(args.epochs):
        # train(model, scaler, device, train_loader, optimizer, epoch) # TODO: clean this
        train(model, device, train_loader, optimizer, epoch, sdmp_args.rank)
        scheduler.step()

    if sdmp_args.rank == 0:
        test(model, device, test_loader)

    # Waiting the save checkpoint to be finished before run another allgather_object
    smp.barrier()

    if sdmp_args.rank == 0:
        model_dict = model.local_state_dict()
        opt_dict = optimizer.local_state_dict()
        smp.save(
            {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
            os.path.join(os.getenv("SM_MODEL_DIR"), "model_checkpoint.pt"),
            partial=True,
        )
    smp.barrier()

    # if smp.local_rank() == 0:
    #    print("Start syncing")
    #    curr_host = os.getenv("SM_CURRENT_HOST")
    #    full_s3_path = f"{args.sync_s3_path}/checkpoints/{curr_host}/"
    #    sync_local_checkpoints_to_s3(
    #        local_path="/opt/ml/local_checkpoints", s3_path=full_s3_path
    #    )
    #    print("Finished syncing")


if __name__ == "__main__":
    main()
