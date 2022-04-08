from __future__ import division, print_function

# Common imports
import argparse
import os
import random

# Third Party imports
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms

BACKEND = "nccl"

# Initialize Distributed Training group
# Note, that we are using env vars setup by torch.distirbuted.run utility before:
# See details: https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py#L208
# reference to api: https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#init_process_group
dist.init_process_group(
    backend=BACKEND,
    rank=int(os.getenv("RANK", 0)),
    world_size=int(os.getenv("WORLD_SIZE", 1)),
)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and args.rank == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * args.world_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        if args.verbose:
            print("Batch", batch_idx, "from rank", args.rank)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # TODO: check if this is applicable for vanilla DDP?
            # SM Distributed: Moves input tensors to the GPU ID used
            # by the current process based on the set_device call.
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def save_model(model, model_dir):
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model.module.state_dict(), f)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement.
    # Each of these variables is model specific.
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


def main():
    print(f"Envvar inside training script: {os.environ}")
    print("Inside main of training script")

    print(f"world size: {dist.get_world_size()}")
    print(f"global rank: {dist.get_rank()}")
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
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--feature-extract", type=bool, default=True)
    parser.add_argument("--use-pretrained", type=bool, default=True)
    parser.add_argument("--sync-s3-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="squeezenet")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="For displaying SMDataParallel-specific logs",
    )
    # Model checkpoint location
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args, unknown = parser.parse_known_args()

    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = int(os.getenv("LOCAL_RANK"))
    args.lr = 1.0

    print(
        f"Collected args: {args}. Following args are not parsed and won't be used: {unknown}."
    )

    # TODO: figure out scaling of batch size
    # args.batch_size //= args.world_size // 8
    # args.batch_size = max(args.batch_size, 1)

    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            args.world_size,
        )

    if not torch.cuda.is_available():
        # TODO: replace with proper message
        raise Exception(
            "Must run SMDataParallel MNIST example on CUDA-capable devices."
        )

    # Initialize the model for this run
    model, input_size = initialize_model(
        args.model_name,
        args.num_classes,
        args.feature_extract,
        use_pretrained=args.use_pretrained,
    )

    # SM Distributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    # Fix seeds in order to get the same losses across runs
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    model = DDP(model.to(device))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

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

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        image_datasets["train"], num_replicas=args.world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            image_datasets["val"], batch_size=args.test_batch_size, shuffle=True
        )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
    if rank == 0:
        test(model, device, test_loader)
    scheduler.step()

    if rank == 0:
        print("Saving the model...")
        save_model(model, args.model_dir)
    dist.barrier()  # putting barrier to avoid open bug: https://github.com/pytorch/pytorch/issues/69520


if __name__ == "__main__":
    main()
