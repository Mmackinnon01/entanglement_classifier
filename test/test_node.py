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

    def test_node_id_assignment(self):
        self.assertEqual(Node(node_id=1).node_id, 1, "Incorrect node id initialised")

    def test_node_creation_without_state(self):
        self.assertTrue((Node(node_id=1).init_quantum_state == spin_down).all(), "Node not initialised in spin down")

    def test_node_creation_in_down_state(self):
        self.assertTrue((Node(node_id=1, init_quantum_state=0).init_quantum_state == spin_down).all(), "Node not initialised in spin down")

    def test_node_creation_in_up_state(self):
        self.assertTrue((Node(node_id=1, init_quantum_state=1).init_quantum_state == spin_up).all(), "Node not initialised in spin up")

    def test_node_creation_in_random_state(self):
        self.assertTrue(((Node(node_id=1, init_quantum_state="random").init_quantum_state == spin_up).all() or (Node(node_id=1, init_quantum_state="random").init_quantum_state == spin_down).all()), "Node not initialised in spin state")

    def test_node_state_change_to_up(self):
        node = Node(node_id=1)
        node.init_quantum_state = 1
        self.assertTrue((node.init_quantum_state == spin_up).all(), "Node not set to spin up")

    def test_node_state_change_to_down(self):
        node = Node(node_id=1, init_quantum_state=1)
        node.init_quantum_state = 0
        self.assertTrue((node.init_quantum_state == spin_down).all(), "Node not set to spin down")
