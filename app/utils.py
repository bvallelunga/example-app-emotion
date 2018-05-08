import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from keras.preprocessing.image import img_to_array


def preprocess(img, target_size):
    """Preprocess the image for model prediction."""
    # Crop img to square with dim = min(height, width)
    half_min_dim = min(img.size) / 2
    half_width, half_height = map(lambda dim: dim / 2, img.size)

    img = img.crop(
        (
            half_width - half_min_dim,
            half_height - half_min_dim,
            half_width + half_min_dim,
            half_height + half_min_dim
        )
    )

    # Greyscale and resize
    img = img.convert('L').resize(target_size)

    # Feature scale image using the same values used at training time
    img = (img_to_array(img) / 255. - .5) * 2.

    # Add batch dim i.e. model expects arrays of shape (batch, height, width, channels)
    return np.expand_dims(img, 0)


def base64_to_img(string):
    """Convert a base64 encoded string to an image."""
    img_buffer = BytesIO(base64.b64decode(string.encode(), validate=True))
    return Image.open(img_buffer)


def get_img(url):
    """Get the image at the provided url."""
    return Image.open(BytesIO(requests.get(url).content))
