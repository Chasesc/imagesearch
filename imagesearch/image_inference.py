'''
Handles ML inference of images. (extract features, class, etc)
'''

from tensorflow.keras import applications

models = {
    'mobilenet' : dict(module=applications.mobilenet,
                       model='MobileNet',
                       features_layer='global_average_pooling2d',
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

        self.decode_predictions = getattr(self.model_module, 'decode_predictions')
        self.preprocess_input   = getattr(self.model_module, 'preprocess_input')
        self.model              = getattr(self.model_module, model_info['model'])(*args, **kwargs)

        if not quiet: self.model.summary()