'''
Handles ML inference of images. (extract features, class, etc)
'''

import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications

models = {
    'mobilenet' : dict(module=applications.mobilenet,
                       model='MobileNet',
                       features_layer='global_average_pooling2d',
                       shape=(224, 224, 3),
                       args=(),
                       kwargs=dict()),

    'xception' : dict(module=applications.xception,
                       model='Xception',
                       features_layer='global_average_pooling2d',
                       shape=(299, 299, 3),
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

        self.decode_predictions = getattr(self.model_module, 'decode_predictions')
        self.preprocess_input   = getattr(self.model_module, 'preprocess_input')
        self.model              = getattr(self.model_module, model_info['model'])(*args, **kwargs)

        if not quiet: self.model.summary()

    def predict(self, img):
        # apply transforms
        img = image.load_img(img, target_size=self.image_shape)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess_input(img)

        preds = self.decode_predictions(self.model.predict(img), top=3)[0]
        return preds