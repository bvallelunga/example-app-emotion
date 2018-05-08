import binascii
import os

import keras.models

from .utils import base64_to_img
from .utils import get_img
from .utils import preprocess

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'fer2013_mini_XCEPTION.102-0.66.hdf5')
LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
SCORE_PRECISION = 2


class ModelInterface(object):
    def __init__(self, model=None):
        if model is None:
            model = keras.models.load_model(MODEL_PATH)
        self.model = model

    def predict(self, input):
        if not (('image_url' in input) ^ ('image_base64' in input)):
            raise KeyError("The input must either have an 'image_url or 'image_base64' key.")
        input_key = 'image_url' if 'image_url' in input else 'image_base64'
        if not isinstance(input_key, str):
            raise ValueError("'{}' must be a string.".format(input_key))
        if input_key == 'image_base64':
            try:
                img = base64_to_img(input[input_key])
            except (binascii.Error, AttributeError):
                raise ValueError("'{}' must be a valid base64 encoded image.".format(input_key))
        else:
            try:
                img = get_img(input[input_key])
            except:
                raise ValueError("'{}' must be a valid image url.".format(input_key))
        min_dim = self.model.input_shape[1]
        if any(dim < min_dim for dim in img.size):
            raise ValueError("'image' can not have a height or width less than {} pixels.".format(min_dim))

        img = preprocess(img, self.model.input_shape[1:3])
        scores = self.model.predict(img).tolist()[0]
        scores = [round(score, SCORE_PRECISION) for score in scores]
        return {label: score for label, score in zip(LABELS, scores)}

