from __future__ import print_function
import logging
import sys
import pandas as pd

import os
import datasets
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
import datasets
from datasets import Dataset


from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DistilBertConfig,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import sys
import argparse
import os

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=-1)

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument(
        "--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    # parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=float, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    LOGGER.setLevel(LOG_LEVEL)
    datasets.utils.logging.set_verbosity(LOG_LEVEL)
    transformers.utils.logging.set_verbosity(LOG_LEVEL)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # logging input parameters
    LOGGER.info(f"Args: {args}")
    LOGGER.info(f"remainder: {remainder}")

    # Leading Feature Store parquet files to HuggingFace dataset
    df = pd.read_parquet(args.training_dir)
    df["input_ids"] = df["tokenized-text"].astype("string")
    train_dataset = Dataset.from_pandas(df[["input_ids", "label"]])

    # Since Feature Store doesn't support Arrays natively,
    # we need to conver input_ids from String to list of integers
    def string_to_list(example):
        list_of_str = example["input_ids"].strip("][").split(", ")
        example["input_ids"] = [int(el) for el in list_of_str]
        return example

    train_dataset = train_dataset.map(string_to_list)
    print(train_dataset[0])
    print("Completed loading data")

    # Set seed before initializing model.
    set_seed(args.seed)

    config = DistilBertConfig()
    config.num_labels = 2
    config.vocab_size = 30522

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )

    # compute metrics function for binary classification
    # def compute_metrics(pred):
    #     labels = pred.label_ids
    #     preds = pred.predictions.argmax(-1)
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         labels, preds, average="binary"
    #     )
    #     acc = accuracy_score(labels, preds)
    #     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        # per_device_eval_batch_size=args.eval_batch_size,
        # warmup_steps=args.warmup_steps,
        # evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
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
        # compute_metrics=compute_metrics,
        train_dataset=train_dataset,
    )

    # train model
    trainer.train()

    # Saves the model to s3
    # trainer.save_model(args.model_dir)
    model.save_pretrained(os.environ["SM_MODEL_DIR"])


if __name__ == "__main__":
    main()
