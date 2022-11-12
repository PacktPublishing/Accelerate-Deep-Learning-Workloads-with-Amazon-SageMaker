import json
import os

from transformers import pipeline

JSON_CONTENT_TYPE = "application/json"

NLP_TASK = os.getenv("NLP_TASK", "question-answering")
TASK_MODEL_MAPPING = {
    "question-answering": "distilbert-base-uncased-distilled-squad",
    "summarization": "distilbart-cnn-6-6",
}


def model_fn(model_dir):
    model_pipeline = pipeline(
        NLP_TASK,
        model=os.path.join(model_dir, TASK_MODEL_MAPPING[NLP_TASK]),
    )
    return model_pipeline


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print(content_type)
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        Exception("Requested unsupported ContentType in Accept: " + content_type)


def predict_fn(input_data, model_pipeline):
    try:
        if NLP_TASK == "summarization":
            results = model_pipeline(
                input_data["article"], max_length=input_data["max_length"]
            )
        else:
            results = model_pipeline(
                question=input_data["question"], context=input_data["context"]
            )
    except Exception as e:
        return str(e)
    return results


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception("Requested unsupported ContentType in Accept: " + accept)
