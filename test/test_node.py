import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from classifier.components.node import Node

spin_up = np.array([[1, 0],
                    [0, 0]])
spin_down = np.array([[0, 0],
                      [0, 1]])


class TestNode(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNode, self).__init__(*args, **kwargs)

    def test_node_creation_without_state(self):
        self.assertTrue((Node().init_quantum_state == spin_down).all(), "Node not initialised in spin down")