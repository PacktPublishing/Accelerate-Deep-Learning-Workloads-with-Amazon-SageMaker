import argparse
import os
import json
import transformers
# import sklearn


def train(args):
    pass


if __name__ =='__main__':

    print(f"TRANFORMER version: {transformers.__version__}")

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--per-device-train-batch-size', type=int, default=16)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=64)

    parser.add_argument('--logging-steps', type=float, default=100)


    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.
    train(args)