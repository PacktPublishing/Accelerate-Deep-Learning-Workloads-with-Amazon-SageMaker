# This code is based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import division, print_function

import argparse
import ast
import os
import random

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

# Make cudnn deterministic in order to get the same losses across runs.
# The following two lines can be removed if they cause a performance impact.
# For more details, see:
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# SM Distributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, data, target):
    # TODO: not using autocase or any AMP
    # with autocast(1 > 0):
    #    output = model(data)
    output = model(data)
    # original loss from SM primer
    #     loss = F.nll_loss(output, target, reduction="mean")
    loss = F.cross_entropy(output, target)

    # scaled_loss = loss  # TODO: it's not scaling it... need to do something about it
    model.backward(
        loss
    )  #  instead of usual loss.backward(), so SMP can control loss calculations
    return output, loss


def train(model, device, train_loader, optimizer, epoch):
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

        if smp.rank() == 0 and batch_idx % 10 == 0:
            print(
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
    if smp.mp_rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss


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

    elif model_name == "inception":
        """Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        raise ValueError(f"Invalid model name {model_name}, exiting...")
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


def main():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feature-extract", type=bool, default=True)
    parser.add_argument("--use-pretrained", type=bool, default=True)
    parser.add_argument("--sync-s3-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="squeezenet")
    parser.add_argument(
        "--mp_parameters", type=str, default="", help="Dictionary with SDT parameters"
    )
    args, unknown = parser.parse_known_args()

    print(
        f"Collected args: {args}. Following args are not parsed and won't be used: {unknown}."
    )

    # this is parsing MP parameters passed as string by SageMaker
    # E.g. "--mp_parameters","auto_partition=True,ddp=False,microbatches=8,optimize=speed,partitions=2,pipeline=interleaved,placement_strategy=cluster"
    sdt_params_strings = dict(item.split("=") for item in args.mp_parameters.split(","))
    sdt_params = _cast_dict_from_string(sdt_params_strings)
    print(f"Parsed SDT params are: {sdt_params}")

    if not torch.cuda.is_available():
        raise ValueError("The script requires CUDA support, but CUDA not available")

    # Initialize the model for this run
    model, input_size = initialize_model(
        args.model_name,
        args.num_classes,
        args.feature_extract,
        use_pretrained=args.use_pretrained,
    )
    # Print the model we just instantiated
    print(model)

    # SM Distributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    # Fix seeds in order to get the same losses across runs
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    smp.init()

    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")
    dataloader_kwargs = {"batch_size": args.batch_size}
    dataloader_kwargs.update(
        {"num_workers": 1, "pin_memory": True, "shuffle": False, "drop_last": True}
    )

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
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

    print(f"SMP DP size:{smp.dp_size()}")
    print(f"SMP DP rank:{smp.dp_rank()}")
    if (sdt_params.ddp) and smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        dataset_train = SplitDataset(
            image_datasets["train"], partitions=partitions_dict
        )
        dataset_train.select(f"{smp.dp_rank()}")
        dataset_val = SplitDataset(image_datasets["val"], partitions=partitions_dict)
        dataset_val.select(f"{smp.dp_rank()}")

    elif (not sdt_params.ddp) or smp.dp_size() == 1:
        dataset_train = image_datasets["train"]
        dataset_val = image_datasets["val"]
    else:
        raise Exception("Check your distributed configuration.")

    train_loader = torch.utils.data.DataLoader(dataset_train, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_val, **dataloader_kwargs)

    optimizer = optim.Adadelta(model.parameters(), lr=4.0)
    model = smp.DistributedModel(model)
    # scaler = smp.amp.GradScaler()  # TODO: remove this as we are not using AMP
    optimizer = smp.DistributedOptimizer(optimizer)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(args.epochs):
        # train(model, scaler, device, train_loader, optimizer, epoch) # TODO: clean this
        train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        print(f"Test loss={test_loss} at {epoch} epoch.")
        scheduler.step()

    # TODO: this needs to be removed.
    # if smp.rank() == 0:
    #    if os.path.exists("/opt/ml/local_checkpoints"):
    #        print("-INFO- PATH DO EXIST")
    #    else:
    #        os.makedirs("/opt/ml/local_checkpoints")
    #        print("-INFO- PATH DO NOT EXIST")

    # Waiting the save checkpoint to be finished before run another allgather_object
    smp.barrier()

    if smp.dp_rank() == 0:
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
