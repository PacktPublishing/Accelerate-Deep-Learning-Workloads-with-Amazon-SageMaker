from transformers import (
    pipeline,
    DistilBertTokenizerFast,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)
import torch
import json


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6 # number of categories
MAX_LENGTH = 512 # max number of tokens model can handle


def model_fn(model_dir):
    """
    Load required components (model, config and tokenizer) to constuct inference pipeline.

    This method is executed only once when SageMaker starts model server.
    """

    # If CUDA device is present, then use it for inference
    # otherwise fallback to CPU
    device_id = 0 if torch.cuda.is_available() else -1

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    config = DistilBertConfig()
    config.num_labels = NUM_LABELS

    model = DistilBertForSequenceClassification.from_pretrained(
        model_dir, config=config
    )

    inference_pipeline = pipeline(
        model=model,
        task="text-classification",
        tokenizer=tokenizer,
        framework="pt",
        device=device_id,
        max_length=MAX_LENGTH,
        truncation=True
    )

    return inference_pipeline


def transform_fn(inference_pipeline, data, content_type, accept_type):
    """
    Deserialize inference request payload, run inferenece, and return back serialized response.
    Note, that currently only JSON is supported, however, this can be extended further as needed.

    This method is executed on every request to SageMaker endpoint.
    """

    # Deserialize payload
    if "json" in content_type:
        deser_data = json.loads(data)
    else:
        raise NotImplemented("Only 'application/json' content type is implemented.")
    
    # Run inference
    predictions = inference_pipeline(deser_data)
    
    # Serialize response
    if "json" in accept_type:
        return json.dumps(predictions)
    else:
        raise NotImplemented("Only 'application/json' accept type is implemented.")
