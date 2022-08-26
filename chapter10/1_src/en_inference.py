import torch
import os
import json
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir):
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print(content_type)
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        Exception("Requested unsupported ContentType in Accept: " + content_type)


def predict_fn(input_data, model_tokenizer_tuple):

    model, tokenizer = model_tokenizer_tuple
    inputs = tokenizer(input_data, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predictions = model.config.id2label[predicted_class_id]
    return predictions


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception("Requested unsupported ContentType in Accept: " + accept)
