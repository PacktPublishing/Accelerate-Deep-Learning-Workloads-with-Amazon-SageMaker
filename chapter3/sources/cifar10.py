import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import json
import cv2
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _get_data_loader(args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=os.getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    return trainloader


def train(args):

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader = _get_data_loader(args)

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(net.state_dict(), os.path.join(os.getenv("SM_MODEL_DIR"), "model.pth"))


def model_fn(model_dir):
    logger.info("Insider model loader")

    model = Net()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    try:
        with open(os.path.join(model_dir, "model.pth"), "rb") as f:
            logger.info(
                f"Trying to open model in {os.path.join(model_dir, 'model.pth')}"
            )
            model.load_state_dict(torch.load(f))
            return model.to(DEVICE)
    except Exception as e:
        logger.exception(e)
        return None


def _load_from_bytearray(request_body):
    npimg = np.frombuffer(request_body, np.float32).reshape((1, 3, 32, 32))
    return torch.Tensor(npimg)


def transform_fn(model, request_body, content_type, accept_type):

    logger.info("Running inference inside container")

    try:
        np_image = _load_from_bytearray(request_body)
        logger.info("Deserialization completed")
    except Exception as e:
        logger.exception(e)

    logger.info("trying to run inference")
    try:
        outputs = model(np_image)
        _, predicted = torch.max(outputs, 1)
        logger.info(f"Predictions: {predicted}")
    except Exception as e:
        logger.exception(e)

    return json.dumps(predicted.numpy().tolist())


if __name__ == "__main__":

    # SageMaker passes hyperparameters  as command-line arguments to the script
    # Parsing them below...

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args, _ = parser.parse_known_args()

    train(args)
