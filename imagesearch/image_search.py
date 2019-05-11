'''
'''
from pathlib import Path
from skimage.io import imread
import numpy as np

import image_inference
import image_utils

class ImageSearch(object):
    def __init__(self, cache_images=True):
        self.images = {}

        self._inference = image_inference.ImageInference()
        self._cache_images = cache_images

    def find_similar(self, img_path, method='features'):
        if img_path not in self.images:
            self.add_images(img_path)

        if method == 'features':
            return self._find_similar_features(img_path)
        else:
            raise ValueError(f'Method "{method}" not yet supported')        

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

            self.images[image] = image_info

    def add_directories(self, *directories, filetypes=('jpg', 'jpeg', 'png'), recursive=False):
        images = []

        for directory in directories:
            path = Path(directory)
            glob = path.rglob if recursive else path.glob

            for filetype in filetypes:
                images.extend(glob(f'*.{filetype}'))

        self.add_images(*images)

    def _find_similar_features(self, img_path):
        distances = []
        for path in self.images:
            if path != img_path:
                dist = self._feature_distance(img_path, path)
                distances.append((dist, path))

        return sorted(distances, key=lambda t:t[0])

    def _feature_distance(self, first_img_path, second_img_path):
        first_img_features  = self.images[first_img_path]['features']
        second_img_features = self.images[second_img_path]['features']
        return np.linalg.norm(second_img_features - first_img_features)

def test():
    imgsearch = ImageSearch()
    imgsearch.add_directories('/home/chases/Pictures/', recursive=True)

if __name__ == '__main__':
    test()
