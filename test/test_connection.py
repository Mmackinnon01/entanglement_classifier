import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from classifier.components.connection import ConnectionFactory


def test_function(a, b, c, model_state):
    return a + b + c + model_state


class TestConnection(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestConnection, self).__init__(*args, **kwargs)

    def test_connection_with_preset_equation(self):
        self.assertEqual(
            ConnectionFactory(test_function, a=1, b=1, c=1)
            .generateConnection(1, 2, 4)
            .calc(model_state=0),
            3,
            "Incorrect function output",
        )

    def test_connection_with_random_variables(self):
        self.assertTrue(
            isinstance(
                ConnectionFactory(test_function, a=[0, 1], b=[0, 1], c=[0, 1])
                .generateConnection(1, 2, 4)
                .calc(model_state=0),
                float,
            ),
            "Incorrect function output type",
        )
