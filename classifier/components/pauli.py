import numpy as np
from sympy.physics.quantum import TensorProduct

sigma_plus = np.array([[0, 1], [0, 0]])
sigma_minus = np.array([[0, 0], [1, 0]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
identity = np.array([[1, 0], [0, 1]])


def sigmaPlus(node, n_nodes):
    total_state = 0
    for n in range(n_nodes):
        if n == node:
            next_state = sigma_plus
        else:
            next_state = identity

        if n == 0:
            total_state = next_state
        else:
            total_state = TensorProduct(total_state, next_state)
    return total_state


def sigmaMinus(node, n_nodes):
    total_state = 0
    for n in range(n_nodes):
        if n == node:
            next_state = sigma_minus
        else:
            next_state = identity

        if n == 0:
            total_state = next_state
        else:
            total_state = TensorProduct(total_state, next_state)
    return total_state


def sigmaX(node, n_nodes):
    total_state = 0
    for n in range(n_nodes):
        if n == node:
            next_state = sigma_x
        else:
            next_state = identity

        if n == 0:
            total_state = next_state
        else:
            total_state = TensorProduct(total_state, next_state)
    return total_state


def sigmaY(node, n_nodes):
    total_state = 0
    for n in range(n_nodes):
        if n == node:
            next_state = sigma_y
        else:
            next_state = identity

        if n == 0:
            total_state = next_state
        else:
            total_state = TensorProduct(total_state, next_state)
    return total_state


def sigmaXMulti(nodes, n_nodes):
    total_state = 0
    for n in range(n_nodes):
        if n in nodes:
            next_state = sigma_x
        else:
            next_state = identity

        if n == 0:
            total_state = next_state
        else:
            total_state = TensorProduct(total_state, next_state)
    return total_state


def sigmaYMulti(nodes, n_nodes):
    total_state = 0
    for n in range(n_nodes):
        if n in nodes:
            next_state = sigma_y
        else:
            next_state = identity

        if n == 0:
            total_state = next_state
        else:
            total_state = TensorProduct(total_state, next_state)
    return total_state
