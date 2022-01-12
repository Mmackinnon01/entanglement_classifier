import unittest
import numpy as np
import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from classifier.components.connection import ConnectionFactory
from classifier.components.reservoir import Reservoir

spin_up = np.array([[1, 0], [0, 0]])
spin_down = np.array([[0, 0], [0, 1]])


def test_function(a, b, c, model_state):
    return a + b + c + model_state


class TestReservoir(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestReservoir, self).__init__(*args, **kwargs)
        self.factory = ConnectionFactory(test_function, a=1, b=1, c=1)

    def test_reservoir_number_nodes_setup(self):
        res = Reservoir(self.factory)
        res.setupNodes(10, system_nodes=3)
        self.assertEqual(len(res.nodes), 10, "Incorrect number of nodes created")

    def test_reservoir_spin_down_nodes_setup(self):
        res = Reservoir(self.factory)
        res.setupNodes(1, system_nodes=3)
        self.assertTrue(
            (res.nodes[3].init_quantum_state == spin_down).all(),
            "Node not initialised in spin down",
        )

    def test_reservoir_spin_up_nodes_setup(self):
        res = Reservoir(self.factory)
        res.setupNodes(1, system_nodes=3, quantum_state=1)
        self.assertTrue(
            (res.nodes[3].init_quantum_state == spin_up).all(),
            "Node not initialised in spin up",
        )

    def test_reservoir_number_connection_setup(self):
        res = Reservoir(self.factory)
        res.setupNodes(10, system_nodes=3)
        res.setupConnections()
        self.assertEqual(
            len(res.connections), 9 * 5, "Incorrect number of connections created"
        )

    def test_reservoir_connection_node_ids(self):
        res = Reservoir(self.factory)
        res.setupNodes(3, system_nodes=3)
        res.setupConnections()
        self.assertTrue(
            (
                res.connections["res_connection_34"].node1 == 3
                and res.connections["res_connection_34"].node2 == 4
            ),
            "Incorrect node assignment",
        )

    def test_reservoir_connection_equation(self):
        res = Reservoir(self.factory)
        res.setupNodes(2, system_nodes=3)
        res.setupConnections()
        self.assertEqual(
            res.connections["res_connection_34"].calc(model_state=0),
            3,
            "Incorrect connection calc",
        )
