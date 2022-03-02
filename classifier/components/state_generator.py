import numpy as np
from sympy.physics.quantum import TensorProduct


def dagger(state):
    return np.conjugate(np.transpose(state))


def generatePureState(dim=2):
    state = np.random.normal(size=(dim, 1))
    norm_const = 0
    for val in state:
        norm_const += val**2
    norm_const = norm_const**0.5
    state = state/norm_const
    density_matrix = TensorProduct(state, dagger(state))
    return density_matrix


def generateMixedState(dim=2):
    T = np.random.normal(size=(dim, dim)) + 1j * \
        np.random.normal(size=(dim, dim))

    density_matrix = np.matmul(T, dagger(T))/np.trace(np.matmul(T, dagger(T)))
    return density_matrix


def generatePureBatch(n_states, dim=2):
    return [generatePureState(dim) for n in range(n_states)]


def generateMixedBatch(n_states, dim=2):
    return [generateMixedState(dim) for n in range(n_states)]
