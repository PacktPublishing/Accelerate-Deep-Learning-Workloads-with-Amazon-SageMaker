from tensorflow.keras.applications.vgg16 import VGG16

class KerasModel(object):
    """
    Sample Keras model loader for illustrative purposes.
    """
    
    @staticmethod
    def get_model(self, model_name="vgg16"):
        
        if model_name.lower()=="vgg16":
            return VGG16(weights='imagenet')
        else:
            raise ValueError(f"model name {model_name} is not supported.")