import numpy as np
from numpy.linalg import multi_dot

from .pauli import sigmaPlus, sigmaMinus, sigmaX, sigmaY, sigmaXMulti, sigmaYMulti


def commutator(a, b):
    return np.matmul(a, b) - np.matmul(b, a)


class DrivenCascadeFunction:
    def __init__(self, node1, node2, n_nodes, gamma_1, gamma_2):
        self.node1 = node1
        self.node2 = node2
        self.n_nodes = n_nodes
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sigma_plus_1 = sigmaPlus(node1, n_nodes)
        self.sigma_plus_2 = sigmaPlus(node2, n_nodes)
        self.sigma_minus_1 = sigmaMinus(node1, n_nodes)
        self.sigma_minus_2 = sigmaMinus(node2, n_nodes)

    def calc(self, model_state):

        term1 = (self.gamma_1) * (
            2 * multi_dot([self.sigma_minus_1, model_state, self.sigma_plus_1])
            - multi_dot([model_state, self.sigma_plus_1, self.sigma_minus_1])
            - multi_dot([self.sigma_plus_1, self.sigma_minus_1, model_state])
        )
        term2 = (self.gamma_2) * (
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


class EnergyExchangeFunction:
    def __init__(self, node1, node2, n_nodes, J):
        self.node1 = node1
        self.node2 = node2
        self.n_nodes = n_nodes
        self.J = J
        self.sigma_x_12 = sigmaXMulti([node1, node2], n_nodes)
        self.sigma_y_12 = sigmaYMulti([node1, node2], n_nodes)

    def calc(self, model_state):
        H = self.J * (self.sigma_x_12 + self.sigma_y_12)
        return H
