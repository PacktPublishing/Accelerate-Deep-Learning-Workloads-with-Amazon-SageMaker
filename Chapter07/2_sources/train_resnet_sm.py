import argparse
import logging
import os
import time

import smdebug.pytorch as smd
import torch
import torch.nn as nn
import torch.optim as optim
from smdebug import modes
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 2  # two classes: ants and bees


def model_step(model, data, target, criterion):
    outputs = model(data)
    loss = criterion(outputs, target)
    return outputs, loss


def train_model(
    model, device, dataloaders, criterion, optimizer, scheduler, args, hook
):

    for epoch in range(1, args.num_epochs + 1):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                if hook:
                    hook.set_mode(modes.TRAIN)
            else:
                model.eval()  # Set model to evaluate mode
                if hook:
                    hook.set_mode(modes.EVAL)

            running_corrects = 0
            running_loss = 0.0
            step_counter = 0
            epoch_start = time.time()

            for _, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs, step_loss = model_step(model, inputs, labels, criterion)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        step_loss.backward()
                        optimizer.step()

                running_corrects += torch.sum(preds == labels.data)
                running_loss += step_loss.item() * inputs.size(0)
                step_counter += 1
            if phase == "train":
                scheduler.step()
            epoch_time = time.time() - epoch_start
            epoch_accuracy = running_corrects.double() / (
                args.batch_size * step_counter
            )
            epoch_loss = running_loss / (args.batch_size * step_counter)
            LOGGER.info(
                f"Epoch {epoch}: {phase} loss {epoch_loss}, accuracy {epoch_accuracy},  in {epoch_time} sec. Total number of steps: {step_counter}"
            )


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_resnet_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-extract", type=bool, default=False)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-data-workers", type=int, default=4)
    return parser.parse_known_args()


def get_dataloaders(args):
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

    LOGGER.info("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.environ[f"SM_CHANNEL_{x.upper()}"], data_transforms[x]
        )
        for x in ["train", "val"]
    }
    dataloaders_dict = {}
    dataloaders_dict["train"] = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=args.batch_size,
        num_workers=args.num_data_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    dataloaders_dict["val"] = torch.utils.data.DataLoader(
        image_datasets["val"],
        batch_size=args.batch_size,
        num_workers=args.num_data_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    return dataloaders_dict


def main():

    args, unknown_args = parse_args()
    LOGGER.info(
        f"Collected hyperparameters: {args}."
        f"Following args are not parsed correctly and won't be used: {unknown_args}."
    )

    if not torch.cuda.is_available():
        raise ValueError("The script requires CUDA support, but CUDA not available")

    print(f"Device count: {torch.cuda.device_count()}")
    model = initialize_resnet_model(
        NUM_CLASSES, feature_extract=False, use_pretrained=True
    )

    device = torch.device("cuda")
    model.to(device)

    on_cuda = next(model.parameters()).is_cuda
    if not on_cuda:
        print(f"Model is not on cuda. On_cuda is {on_cuda}. Trying something else")
        new_device = torch.device("cuda:0")
        torch.cuda.set_device(new_device)
        model.cuda()
        on_cuda = next(model.parameters()).is_cuda
        if not on_cuda:
            raise Exception(f"Model is not on cuda. On_cuda is {on_cuda}. We're done")

    params_to_update = model.parameters()
    info_message = "Params to learn:"
    if args.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                info_message += f"{name}, "
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                info_message += f"{name}, "
    LOGGER.info(info_message)

    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize Hook object
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)

    dataloaders_dict = get_dataloaders(args)
    model = train_model(
        model,
        device,
        dataloaders_dict,
        criterion,
        optimizer,
        exp_lr_scheduler,
        args,
        hook,
    )


if __name__ == "__main__":
    main()
