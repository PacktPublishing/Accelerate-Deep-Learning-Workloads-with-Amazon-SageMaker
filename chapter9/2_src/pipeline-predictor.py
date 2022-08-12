import torch
import os
import json
from transformers import pipeline


JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir):
    print(f"!!!!!!!!! loading model from {model_dir}")
    model_pipeline = pipeline(
        "question-answering",
        model=os.path.join(model_dir, "distilbert-base-uncased-distilled-squad"),
    )
    return model_pipeline


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print("****************")
    print(content_type)
    if content_type == JSON_CONTENT_TYPE:
        # input_data = json.loads(serialized_input_data)
        print("inside deser if")
        print(type(serialized_input_data))
        input_data = json.loads(serialized_input_data)
        print(f"!!!!!!!!! deser input data {input_data}")
        return input_data
    else:
        Exception("Requested unsupported ContentType in Accept: " + content_type)


def predict_fn(input_data, model_pipeline):
    print("Inside predict_fn")
    print(input_data)
    try:
        results = model_pipeline(
            question=input_data["question"], context=input_data["context"]
        )
    except Exception as e:
        print(e)
        return str(e)
    print(f"!!!!! results {results}")
    return results


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    print(f"!!!!! Accept {accept}")
    print(f"!!!!! Pred output {prediction_output}")

    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception("Requested unsupported ContentType in Accept: " + accept)
