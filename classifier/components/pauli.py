import numpy as np
from itertools import product
from sympy.physics.quantum import TensorProduct

sigma_plus = np.array([[0, 1], [0, 0]])
sigma_minus = np.array([[0, 0], [1, 0]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
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


def sigmaXExcitationOperator(node, n_nodes):
    total_state = 0
    for n in np.arange(n_nodes):
        if n == node:
            next_operator = sigma_x
        else:
            next_operator = identity

        if n == 0:
            total_state = next_operator
        else:
            total_state = TensorProduct(total_state, next_operator)
    return total_state


def sigmaYExcitationOperator(node, n_nodes):
    total_state = 0
    for n in np.arange(n_nodes):
        if n == node:
            next_operator = sigma_z
        else:
            next_operator = identity

        if n == 0:
            total_state = next_operator
        else:
            total_state = TensorProduct(total_state, next_operator)
    return total_state


def sigmaZExcitationOperator(node, n_nodes):
    total_state = 0
    for n in np.arange(n_nodes):
        if n == node:
            next_operator = sigma_z
        else:
            next_operator = identity

        if n == 0:
            total_state = next_operator
        else:
            total_state = TensorProduct(total_state, next_operator)
    return total_state


def sigmaCombinatorics(n_nodes):
    operators = {}
    for combination in list(product([["x", sigma_x], ["y", sigma_y], ["z", sigma_z]], repeat=n_nodes)):
        name = ""
        first = True
        for sigma in combination:
            name += sigma[0]
            if first:
                first = False
                operator = sigma[1]
            else:
                operator = TensorProduct(operator, sigma[1])
        operators[name] = operator
    return operators
