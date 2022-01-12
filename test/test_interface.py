import unittest
import numpy as np
import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from classifier.components.connection import ConnectionFactory
from classifier.components.interface import Interface


def test_function(a, b, c, model_state):
    return a + b + c + model_state


class TestInterface(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInterface, self).__init__(*args, **kwargs)
        self.factory = ConnectionFactory(test_function, a=1, b=1, c=1)

    def test_interface_number_connection_setup(self):
        itf = Interface(
            sys_nodes=[0, 1], res_nodes=[2, 3], connectionFactory=self.factory
        )
        itf.setupConnections()
        self.assertEqual(
            len(itf.connections), 4, "Incorrect number of connections created"
        )

    def test_interface_connection_node_ids(self):
        itf = Interface(
            sys_nodes=[0, 1], res_nodes=[2, 3], connectionFactory=self.factory
        )
        itf.setupConnections()
        self.assertTrue(
            (
                itf.connections["interface_connection_02"].node1 == 0
                and itf.connections["interface_connection_02"].node2 == 2
            ),
            "Incorrect node assignment",
        )

    def test_interface_connection_equation(self):
        itf = Interface(
            sys_nodes=[0, 1], res_nodes=[2, 3], connectionFactory=self.factory
        )
        itf.setupConnections()
        self.assertEqual(
            itf.connections["interface_connection_02"].calc(model_state=0),
            3,
            "Incorrect connection calc",
        )
