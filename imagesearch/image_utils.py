
from PIL import Image as pil_image
from tensorflow.keras.preprocessing import image


def load(img_path):
    return image.load_img(img_path)

def resize(img, size, method=pil_image.NEAREST):
    return img.resize(size, method)

def pil_to_array(img):
    return image.img_to_array(img)