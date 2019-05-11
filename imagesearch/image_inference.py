'''
Handles ML inference of images. (extract features, class, etc)
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import applications
import numpy as np

import image_utils

models = {
    'mobilenet' : dict(module=applications.mobilenet,
                       model='MobileNet',
                       features_layer='global_average_pooling2d',
                       shape=(224, 224),
                       args=(),
                       kwargs=dict()),

    'xception' : dict(module=applications.xception,
                       model='Xception',
                       features_layer='avg_pool',
                       shape=(299, 299),
                       args=(),
                       kwargs=dict())
}

class ImageInference(object):
    def __init__(self, model='mobilenet', quiet=False):
        if model not in models:
            opts = ', '.join(models.keys())
            raise ValueError(f'model must be one of ({opts})')

        model_info = models[model]

        args   = model_info['args']
        kwargs = model_info['kwargs']

        self.model_module = model_info['module']
        self.image_shape  = model_info['shape']
        features_layer    = model_info['features_layer']

        self.decode_predictions = getattr(self.model_module, 'decode_predictions')
        self.preprocess_input   = getattr(self.model_module, 'preprocess_input')
        model                   = getattr(self.model_module, model_info['model'])(*args, **kwargs)
        
        # Input: Image, Output: Vector of features
        self.features_model   = Model(inputs=model.input, outputs=model.get_layer(features_layer).output)

        features_layer_idx = [l.name for l in model.layers].index(features_layer)
        input_shape = model.layers[features_layer_idx+1].get_input_shape_at(0)

        prediction_model_input = Input(shape=input_shape)

        x = prediction_model_input
        for layer in model.layers[features_layer_idx+1:]:
            x = layer(x)

        # Input: Vector of features, Output: prediction probabilities
        self.prediction_model = Model(inputs=prediction_model_input, outputs=x)

        if not quiet:
            print(self.features_model.summary())
            print('-'*50)
            print(self.prediction_model.summary())      

    def predict(self, img):
        # apply transforms
        img = image_utils.resize(img, self.image_shape)
        img = image_utils.pil_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)

        features = self.features_model.predict(img)

        preds = self.prediction_model.predict(np.expand_dims(features, axis=0)).reshape((1, 1000))
        preds = self.decode_predictions(preds, top=3)[0]

        return preds, features