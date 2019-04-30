'''
'''

import image_inference

from pathlib import Path
from skimage.io import imread

def load_image_from_path(path):
    return imread(path, plugin='matplotlib')[:, :, :3] # just keep rgb

class ImageSearch(object):
    def __init__(self):
        self._inference = image_inference.ImageInference()
        self._images = {}

    def add_images(self, *images):
        for image in images:
            image = str(image) # ensure we have a string and not a pathlib path

            self._images[image] = self._inference.predict(image)
    
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

if __name__ == '__main__':
    test()