'''
'''

from pathlib import Path
from skimage.io import imread

import image_inference
import image_utils

class ImageSearch(object):
    def __init__(self, cache_images=True):
        self._inference = image_inference.ImageInference()
        self._images = {}

        self._cache_images = cache_images

    def add_images(self, *images):
        for image in images:
            image = str(image) # ensure we have a string and not a pathlib path

            pil_image = image_utils.load(image)
            top_three_preds, features = self._inference.predict(pil_image)

            image_info = {
                'predictions' : top_three_preds,
                'features'    : features,
                'image'       : image_utils.pil_to_array(pil_image) if self._cache_images else None
            }

            self._images[image] = image_info

    def add_directories(self, *directories, filetypes=('jpg', 'jpeg', 'png'), recursive=False):
        images = []

        for directory in directories:
            path = Path(directory)
            glob = path.rglob if recursive else path.glob

            for filetype in filetypes:
                images.extend(glob(f'*.{filetype}'))

        self.add_images(*images)

def test():
    imgsearch = ImageSearch()
    imgsearch.add_directories('/home/chases/Pictures/cats')
    a = 1 # for a breakpoint

if __name__ == '__main__':
    test()
