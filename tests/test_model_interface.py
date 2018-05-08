import os
import unittest

import keras.models

from app.main import MODEL_PATH
from app.main import ModelInterface
from app.main import SCORE_PRECISION
from .utils import img_to_base64

MODEL = keras.models.load_model(MODEL_PATH)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_PATH, 'test.jpg')
IMG_BASE64 = img_to_base64(IMG_PATH)
IMG_URL = 'https://s3-us-west-1.amazonaws.com/doppler-production/production/public/gavin_belson.png'


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.interface = ModelInterface(MODEL)

    def tearDown(self):
        self.interface = None

    def test_missing_both_input_keys(self):
        """Only 1 input key should be provided."""
        with self.assertRaises(KeyError):
            self.interface.predict({})

    def test_includes_both_input_keys(self):
        """Only 1 input key should be provided."""
        with self.assertRaises(KeyError):
            self.interface.predict({'image_url': IMG_URL, 'image_base64': IMG_BASE64})

    def test_image_base64_scores_has_all_emotion_keys(self):
        """The scores dict must all the emotions as keys."""
        scores = self.interface.predict({'image_base64': IMG_BASE64})
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        self.assertTrue(all(emotion in scores.keys() for emotion in emotions))

    def test_image_url_scores_has_all_emotion_keys(self):
        """The scores dict must all the emotions as keys."""
        scores = self.interface.predict({'image_url': IMG_URL})
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        self.assertTrue(all(emotion in scores.keys() for emotion in emotions))

    def test_image_base64_scores_are_floats(self):
        """The score values must be floats."""
        scores = self.interface.predict({'image_base64': IMG_BASE64})
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))

    def test_image_url_scores_are_floats(self):
        """The score values must be floats."""
        scores = self.interface.predict({'image_url': IMG_URL})
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))

    def test_image_base64_scores_have_correct_precision(self):
        """The scores must have the correct precision."""
        scores = self.interface.predict({'image_base64': IMG_BASE64})
        self.assertTrue(all(score == round(score, SCORE_PRECISION) for score in scores.values()))

    def test_image_url_scores_have_correct_precision(self):
        """The scores must have the correct precision."""
        scores = self.interface.predict({'image_url': IMG_URL})
        self.assertTrue(all(score == round(score, SCORE_PRECISION) for score in scores.values()))

    def test_image_base64_is_not_a_string(self):
        """'image_base64' must be a string."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image_base64': []})

    def test_image_url_is_not_a_string(self):
        """'image_url' must be a string."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image_url': []})

    def test_image_base64_is_not_base64_encoded(self):
        """'image_base64' must be a base64 encoded string."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image_base64': '<'})

    def test_image_url_is_not_url(self):
        """'image_url' must be a valid image url."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image_url': 'www.google.com'})

    def test_image_base64_is_too_small(self):
        """'image_base64' can not be smaller than the model's expected input size."""
        with self.assertRaises(ValueError):
            img = img_to_base64(os.path.join(BASE_PATH, 'test_small.jpg'))
            self.interface.predict({'image_base64': img})

    def test_image_url_is_too_small(self):
        """'image_base64' can not be smaller than the model's expected input size."""
        with self.assertRaises(ValueError):
            img = img_to_base64(os.path.join(BASE_PATH, 'test_small.jpg'))
            self.interface.predict({'image_base64': img})
