import os
import unittest

import keras.models

from app.main import MODEL_PATH
from app.main import ModelInterface
from app.main import SCORE_PRECISION
from .utils import img_to_base64

MODEL = keras.models.load_model(MODEL_PATH)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG = img_to_base64(os.path.join(BASE_PATH, 'test.jpg'))


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.interface = ModelInterface(MODEL)

    def tearDown(self):
        self.interface = None

    def test_scores_has_all_emotion_keys(self):
        """The scores dict must all the emotions as keys."""
        scores = self.interface.predict({'image': IMG})
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        self.assertTrue(all(emotion in scores.keys() for emotion in emotions))

    def test_scores_are_floats(self):
        """The score values must be floats."""
        scores = self.interface.predict({'image': IMG})
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))

    def test_scores_have_correct_precision(self):
        """The scores must have the correct precision."""
        scores = self.interface.predict({'image': IMG})
        self.assertTrue(all(score == round(score, SCORE_PRECISION) for score in scores.values()))

    def test_image_is_not_a_string(self):
        """'image' must be a string."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image': []})

    def test_image_is_not_base64_encoded(self):
        """'image' must be a base64 encoded string."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image': '<'})

    def test_image_is_not_url(self):
        """'image' must be a valid image url."""
        with self.assertRaises(ValueError):
            self.interface.predict({'image': 'www.google.com'})

    def test_image_is_too_small(self):
        """'image' can not be smaller than the model's expected input size."""
        with self.assertRaises(ValueError):
            img = img_to_base64(os.path.join(BASE_PATH, 'test_small.jpg'))
            self.interface.predict({'image': img})
