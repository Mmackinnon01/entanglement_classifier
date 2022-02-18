import numpy as np

from .pauli import sigmaX, sigmaXExcitationOperator, sigmaYExcitationOperator, sigmaZExcitationOperator, sigmaXMulti, sigmaYMulti, sigmaCombinatorics


def measureAllExcitations(density_matrix):
    excitations = []
    for system in np.arange(np.log2(density_matrix.shape[0])):
        excitation_val = measureExcitation(density_matrix, system)
        excitations.append(excitation_val)
    return excitations


def measureExcitation(density_matrix, system):
    operator = sigmaZExcitationOperator(
        system, np.log2(density_matrix.shape[0]))
    return np.trace(np.matmul(density_matrix, operator))


def measureTotalExcitations(density_matrix):
    n_nodes = int(
        np.log2(density_matrix.shape[0]))
    x_operator = sigmaXMulti([node for node in range(n_nodes)], n_nodes)
    y_operator = sigmaYMulti([node for node in range(n_nodes)], n_nodes)
    return np.trace(np.matmul(density_matrix, x_operator)), np.trace(np.matmul(density_matrix, y_operator))


def measureAllSigmaCombinations(density_matrix):
    n_nodes = int(
        np.log2(density_matrix.shape[0]))
    operators = sigmaCombinatorics(n_nodes)
    for name, operator in operators.items():
        operators[name] = np.trace(np.matmul(density_matrix, operator))
    return operators
