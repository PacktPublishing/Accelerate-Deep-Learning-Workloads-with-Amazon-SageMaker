import argparse
import os
import json
import torch
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def _get_tokenized_data():
    """
    Read data from CSV files and tokenize datasets for training
    """
    train_dataset = pd.read_csv(
        os.path.join(os.environ["SM_CHANNEL_TRAIN"], "train_dataset.csv"), header=0
    )
    test_dataset = pd.read_csv(
        os.path.join(os.environ["SM_CHANNEL_TEST"], "test_dataset.csv"), header=0
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(
        train_dataset["data"].to_list(), truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_dataset["data"].to_list(), truncation=True, padding=True
    )

    train_enc_dataset = CustomDataset(train_encodings, train_dataset["category_id"])
    test_enc_dataset = CustomDataset(test_encodings, test_dataset["category_id"])

    return train_enc_dataset, test_enc_dataset


def train(args):
    """
    Instantiate tokenizer, model config, and download pretrained model.
    After that run training using hyperparameters defined in SageMaker Training job config.
    If training is succesfull, save trained model.
    """

    train_enc_dataset, test_enc_dataset = _get_tokenized_data()

    training_args = TrainingArguments(
        output_dir=os.getenv(
            "SM_OUTPUT_DIR", "./"
        ),  # output directory, if runtime is not
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
    )

    # Update config for multicategorical task (default is binary classification)
    config = DistilBertConfig()
    config.num_labels = NUM_LABELS

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config
    )

    trainer = Trainer(
        model=model,  # model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_enc_dataset,  # training dataset
        eval_dataset=test_enc_dataset,  # evaluation dataset
    )

    trainer.train()

    # if training is successfuly completed, we save model to SM_MODEL_DIR directory
    # SageMaker will automatically upload any artifacts in this directory to S3
    model.save_pretrained(os.environ["SM_MODEL_DIR"])


if __name__ == "__main__":

    # SageMaker passes hyperparameters  as command-line arguments to the script
    # Parsing them below...
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=float, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    args, _ = parser.parse_known_args()

    train(args)
