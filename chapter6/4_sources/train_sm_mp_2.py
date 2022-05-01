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
import time
import copy

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 2  # two classes: ants and bees

## Make cudnn deterministic in order to get the same losses across runs.
## The following two lines can be removed if they cause a performance impact.
## For more details, see:
## https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# SM Distributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, data, target, criterion):
    output = model(data)
    LOGGER.info(f"outputs from train step: {output}")
    LOGGER.info(f"target from train step: {target}")
    loss = criterion(output, target)
    LOGGER.info(f"loss from train step: {loss}")

    # scaled_loss = loss  # TODO: it's not scaling it... need to do something about it
    model.backward(
        loss
    )  #  instead of usual loss.backward(), so SMP can control loss calculations
    return output, loss


# SM Distributed: Define smp.step for evaluation.
 @smp.step
def test_step(model, data, target, criterion):
    output = model(data)
    loss = criterion(output, target)  # sum up batch loss
    return output, loss


def train_model(model, device, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            LOGGER.info(f"============{phase} phase=========")
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    # backward + optimize only if in training phase
                    if phase == "train":
                        outputs, loss_mb = train_step(model, inputs, labels, criterion)
                        loss = loss_mb.reduce_mean()
                        optimizer.step()
                    else:
                        outputs, loss_mb = test_step(model, inputs, labels, criterion)
                        loss = loss_mb.reduce_mean()
                # LOGGER.info(
                #    f"Outputs from StepOutput should be grouped along microbatch:{outputs.outputs}"
                # )
                # LOGGER.info(f"Concatenated outputs from StepOutput:{outputs.concat()}")
                _, preds = torch.max(outputs.concat(), 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            ## deep copy the model
            # if phase == "val" and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """Densenet"""
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


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
    parser.add_argument("--model-name", type=str, default="squeezenet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-extract", type=bool, default=True)
    parser.add_argument(
        "--mp_parameters", type=str, default="", help="Dictionary with SDT parameters"
    )
    return parser.parse_known_args()


def get_dataloaders(args, sdmp_args):
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

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.environ[f"SM_CHANNEL_{x.upper()}"], data_transforms[x]
        )
        for x in ["train", "val"]
    }
    # Create training and validation dataloaders

    dataloaders_dict = {}
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        image_datasets["train"], num_replicas=sdmp_args.dp_size, rank=sdmp_args.dp_rank
    )

    dataloaders_dict["train"] = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    dataloaders_dict["val"] = torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloaders_dict


def main():

    smp.init()
    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")

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
    model, args.input_size = initialize_model(
        args.model_name, NUM_CLASSES, args.feature_extract, use_pretrained=True
    )

    params_to_update = model.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model = smp.DistributedModel(model)
    optimizer = smp.DistributedOptimizer(optimizer)

    # Scaling batch size if data parallelism used
    if sdmp_args.dp_size > 1:
        old_batch_size = args.batch_size
        args.batch_size //= sdmp_args.dp_size
        args.batch_size = max(args.batch_size, 1)
        LOGGER.info(f"Scaled batch size from {old_batch_size} to {args.batch_size}.")
    dataloaders_dict = get_dataloaders(args, sdmp_args)

    model, hist = train_model(
        model,
        device,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs=args.epochs,
    )
    # Waiting the save checkpoint to be finished before run another allgather_object
    smp.barrier()

    # if sdmp_args.rank == 0:
    #    model_dict = model.local_state_dict()
    #    opt_dict = optimizer.local_state_dict()
    #    smp.save(
    #        {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
    #        os.path.join(os.getenv("SM_MODEL_DIR"), "model_checkpoint.pt"),
    #        partial=True,
    #    )
    # smp.barrier()

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
