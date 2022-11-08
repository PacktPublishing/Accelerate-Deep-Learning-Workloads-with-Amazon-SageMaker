import json
import os
import sys

# Imports for GRPC invoke on TFS
import grpc
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.compat.v1 import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

HEIGHT = os.getenv("IMAGE_HEIGHT", 224)
WIDTH = os.getenv("IMAGE_WIDTH", 224)
MAX_GRPC_MESSAGE_LENGTH = 512 * 1024 * 1024
USE_GRPC = True if os.getenv("USE_GRPC").lower() == "true" else False

# Restrict memory growth on GPU's
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
if physical_gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("**** NO physical GPUs")


def preprocess_image(image):
    image = np.array(image)
    # reshape into shape [batch_size, height, width, num_channels]
    img_reshaped = tf.reshape(
        image, [1, image.shape[0], image.shape[1], image.shape[2]]
    )
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
    return image


def _predict_using_grpc(context, instance):
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = "model"
    grpc_request.model_spec.signature_name = "serving_default"

    options = [
        ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_LENGTH),
    ]

    channel = grpc.insecure_channel(f"0.0.0.0:{context.grpc_port}", options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    grpc_request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(instance))
    result = stub.Predict(grpc_request, 10)
    output_shape = [dim.size for dim in result.outputs["output_1"].tensor_shape.dim]
    print(f"output shape: {output_shape}")
    np_result = np.array(result.outputs["output_1"].float_val).reshape(output_shape)
    return json.dumps({"predictions": np_result.tolist()})


def handler(data, context):

    if context.request_content_type == "application/json":
        instance = json.loads(data.read().decode("utf-8"))
    else:
        raise ValueError(
            415,
            'Unsupported content type "{}"'.format(
                context.request_content_type or "Unknown"
            ),
        )

    if USE_GRPC:
        prediction = _predict_using_grpc(context, instance)
    else:
        inst_json = json.dumps({"instances": instance})
        response = requests.post(context.rest_uri, data=inst_json)
        if response.status_code != 200:
            raise Exception(response.content.decode("utf-8"))
        res = response.content
        request_size = sys.getsizeof(inst_json)
        response_size = sys.getsizeof(res)
        print("request payload size")
        print(request_size)
        print("response payload size")
        print(response_size)
        prediction = res

    response_content_type = context.accept_header
    return prediction, response_content_type
