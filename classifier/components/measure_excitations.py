import numpy as np

from .pauli import excitationOperator


def measureAllExcitations(density_matrix):
    excitations = []
    for system in np.arange(np.log2(density_matrix.shape[0])):
        excitation_val = measureExcitation(density_matrix, system)
        excitations.append(excitation_val)
    return excitations


def measureExcitation(density_matrix, system):
    operator = excitationOperator(system, np.log2(density_matrix.shape[0]))
    return np.trace(np.matmul(density_matrix, operator))
