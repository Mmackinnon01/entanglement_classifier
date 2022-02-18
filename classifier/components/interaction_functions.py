from sympy.physics.quantum import TensorProduct
import numpy as np
from numpy.linalg import multi_dot

from .pauli import sigmaPlus, sigmaMinus, sigmaX, sigmaY, sigmaXMulti, sigmaYMulti


def commutator(a, b):
    return np.matmul(a, b) - np.matmul(b, a)


class DampingFunction:
    def __init__(self, nodes, n_nodes, damping_strength):
        self.node = nodes
        self.n_nodes = n_nodes
        self.damping_strength = damping_strength
        self.sigma_plus = sigmaPlus(self.node, n_nodes)
        self.sigma_minus = sigmaMinus(self.node, n_nodes)

    def calc(self, ro):
        value = (self.damping_strength / 2) * (
            2 * multi_dot([self.sigma_minus, ro, self.sigma_plus])
            - multi_dot([ro, self.sigma_plus, self.sigma_minus])
            - multi_dot([self.sigma_plus, self.sigma_minus, ro])
        )
        return value


class CascadeFunction:
    def __init__(self, nodes, n_nodes, gamma_1, gamma_2):
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.n_nodes = n_nodes
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sigma_plus_1 = sigmaPlus(self.node1, n_nodes)
        self.sigma_plus_2 = sigmaPlus(self.node2, n_nodes)
        self.sigma_minus_1 = sigmaMinus(self.node1, n_nodes)
        self.sigma_minus_2 = sigmaMinus(self.node2, n_nodes)

    def calc(self, ro):
        term1 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * commutator(
            self.sigma_plus_2, np.matmul(self.sigma_minus_1, ro)
        )
        term2 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * commutator(
            np.matmul(ro, self.sigma_plus_1), self.sigma_minus_2
        )
        return term1 + term2


class EnergyExchangeFunction:
    def __init__(self, nodes, n_nodes, coupling_strength):
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.n_nodes = n_nodes
        self.coupling_strength = coupling_strength
        self.sigma_x_12 = sigmaXMulti([self.node1, self.node2], n_nodes)
        self.sigma_y_12 = sigmaYMulti([self.node1, self.node2], n_nodes)

    def calc(self, ro):
        H = self.coupling_strength * (self.sigma_x_12 + self.sigma_y_12)
        value = -1j * commutator(H, ro)
        return value


class DampedCascadeFunction:
    def __init__(self, nodes, n_nodes, gamma_1, gamma_2):
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.n_nodes = n_nodes
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sigma_plus_1 = sigmaPlus(self.node1, n_nodes)
        self.sigma_plus_2 = sigmaPlus(self.node2, n_nodes)
        self.sigma_minus_1 = sigmaMinus(self.node1, n_nodes)
        self.sigma_minus_2 = sigmaMinus(self.node2, n_nodes)

    def calc(self, model_state):

        term1 = (self.gamma_1) * (
            2 * multi_dot([self.sigma_minus_1, model_state, self.sigma_plus_1])
            - multi_dot([model_state, self.sigma_plus_1, self.sigma_minus_1])
            - multi_dot([self.sigma_plus_1, self.sigma_minus_1, model_state])
        )
        term2 = 0.1 * (self.gamma_2) * (
            2 * multi_dot([self.sigma_minus_2, model_state, self.sigma_plus_2])
            - multi_dot([model_state, self.sigma_plus_2, self.sigma_minus_2])
            - multi_dot([self.sigma_plus_2, self.sigma_minus_2, model_state])
        )
        term3 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * commutator(
            self.sigma_plus_2, np.matmul(self.sigma_minus_1, model_state)
        )
        term4 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * commutator(
            np.matmul(model_state, self.sigma_plus_1), self.sigma_minus_2
        )
        return term1 + term2 + term3 + term4
