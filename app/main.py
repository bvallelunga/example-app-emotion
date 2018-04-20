import os
import binascii

import keras.models

from .utils import base64_to_img
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
        if 'image' not in input:
            raise KeyError("No key named 'image' in input.")
        try:
            img = base64_to_img(input['image'])
        except (binascii.Error, AttributeError):
            raise ValueError("'image' must be a base64 encoded string.")
        min_dim = self.model.input_shape[1]
        if any(dim < min_dim for dim in img.size):
            raise ValueError("'image' can not have a height or width less than {} pixels.".format(min_dim))

        img = preprocess(img, self.model.input_shape[1:3])
        scores = self.model.predict(img).tolist()[0]
        scores = [round(score, SCORE_PRECISION) for score in scores]
        return {label: score for label, score in zip(LABELS, scores)}

