from __future__ import division, print_function

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.torch_smddp
import torch.distributed as dist

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms
import logging


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

dist.init_process_group(backend="smddp")

# Two classes: ants and bees
NUM_CLASSES = 2


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
            LOGGER.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * args.world_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        LOGGER.info(f"Batch {batch_idx} from rank {args.rank}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
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
    LOGGER.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def save_model(model, model_dir):
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model.module.state_dict(), f)


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters set by users are passed as command-line arguments to the script.
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
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
    # Model checkpoint location
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    return parser.parse_known_args()


def get_dataloaders(args):
    # Data augmentation and normalization for training
    # Just normalization for validation
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

    # Note that we are passing global rank in data samples to get unique data slice
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        image_datasets["train"], num_replicas=args.world_size, rank=args.rank
    )
    train_loader = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = None
    if args.rank == 0:
        test_loader = torch.utils.data.DataLoader(
            image_datasets["val"], batch_size=args.test_batch_size, shuffle=True
        )
    return train_loader, test_loader


def get_resnet_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    input_size = 224  # defined by Resnet18

    return model_ft, input_size


def main():

    args, unknown_args = parse_args()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    args.local_rank = int(os.getenv("LOCAL_RANK", 1))
    LOGGER.info(
        f"Collected args: {args}. Following args are not parsed and will be ignored: {unknown_args}."
    )

    LOGGER.info(
        f"Hello from training process with rank={args.rank} and "
        f"local rank={args.local_rank} in the world of {args.world_size}"
    )

    if not torch.cuda.is_available():
        raise Exception(
            "Must run SMDataParallel MNIST example on CUDA-capable devices."
        )

    model, args.input_size = get_resnet_model()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")

    model = DDP(model.to(device))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Converting global batch size to local one
    args.batch_size //= dist.get_world_size()
    args.batch_size = max(args.batch_size, 1)
    train_loader, test_loader = get_dataloaders(args)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
    if args.rank == 0:
        test(model, device, test_loader)

    if args.rank == 0:
        LOGGER.info("Saving the model...")
        save_model(model, args.model_dir)


if __name__ == "__main__":
    main()
