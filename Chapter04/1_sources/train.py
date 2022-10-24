import argparse
import logging
import os

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DistilBertConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=float, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args, unknown_args = parser.parse_known_args()

    # Adding parameters of SageMaker training environment
    args.output_data_dir = os.getenv("SM_OUTPUT_DATA_DIR")
    args.model_dir = os.getenv("SM_MODEL_DIR")
    args.training_dir = os.getenv("SM_CHANNEL_TRAINING")

    return args, unknown_args


def main():

    args, _ = parse_args()
    LOGGER.info(f"Training arguments: {args}")

    # Reading Feature Store parquet files to HuggingFace dataset
    df = pd.read_parquet(args.training_dir)
    df["input_ids"] = df["tokenized-text"].astype("string")
    train_dataset = Dataset.from_pandas(df[["input_ids", "label"]])

    # Since Feature Store doesn't support arrays natively, need to conver input_ids from String to list of integers
    def string_to_list(example):
        list_of_str = example["input_ids"].strip("][").split(", ")
        example["input_ids"] = [int(el) for el in list_of_str]
        return example

    train_dataset = train_dataset.map(string_to_list)

    LOGGER.info("Completed loading data")

    # Set seed before initializing model.
    set_seed(args.seed)

    # Intializing models for binary classification
    config = DistilBertConfig()
    config.num_labels = 2
    config.vocab_size = 30522

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=args.learning_rate,
        prediction_loss_only=True,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # train model
    trainer.train()

    # Saves the model to s3
    model.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()
