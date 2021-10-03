from collections import namedtuple
import glob
import json
import logging
import os
import re

import numpy as np
from sagemaker_inference import content_types, encoder


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import decode_predictions
import cv2

from keras_model_loader import KerasModel

IMAGE_SHAPE = (1, 224, 224, 3)

class ModelHandler(object):
    """
    Keras VGG pre-trained model classifier
    """

    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return: None
        """
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir") 
        
        self.model = KerasModel.get_model("vgg16")
        print(self.model.summary())
       

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """        
        payload = request[0]['body']
        
        nparr = np.frombuffer(payload, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = np.asarray(img_np)
        image = np.reshape(image, IMAGE_SHAPE)
        print(f"image type={type(image)}, shape={np.shape(image)}")

        return image

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in numpy array
        """
        prediction = self.model.predict(model_input)
        print(f"prediction={prediction}")
        labels = decode_predictions(prediction)
        print(f"prediction={labels}")
        
        most_likely_label = labels[0][0][1]
        print(f"most_likely_label={most_likely_label}")

        
        return most_likely_label

    def postprocess(self, inference_output):
        """
        Post processing step - converts predictions to str
        :param inference_output: predictions as numpy
        :return: list of inference output as string
        """
        print("inside postproc")
        otput_ser = json.dumps(inference_output)
        return [otput_ser]
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        
        print(model_out)
        
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)