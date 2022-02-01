import numpy as np
from sympy.physics.quantum import TensorProduct


def dagger(state):
    return np.conjugate(np.transpose(state))


def generatePureState():
    a = np.random.rand()
    b = np.random.rand()
    norm_const = (a**2 + b**2)**0.5

    state = np.array([[a], [b]]) / norm_const
    density_matrix = TensorProduct(state, dagger(state))
    return density_matrix


def generateMixedState():
    T11 = np.random.rand()
    T12 = np.random.rand()
    T21 = np.random.rand()
    T22 = np.random.rand()

    T = np.array([[T11, T12], [T21, T22]])

    density_matrix = np.matmul(T, dagger(T))/np.trace(np.matmul(T, dagger(T)))
    return density_matrix


def generateMixedStateUsingPure():
    state = generatePureState()

    alpha = np.random.rand()*0.9

    mixed_state = alpha * state + (1-alpha)/2 * np.identity(state.shape[0])
    return mixed_state


def generatePureBatch(n_states):
    return [generatePureState() for n in range(n_states)]


def generateMixedBatch(n_states):
    return [generateMixedState() for n in range(n_states)]


def generateMixedBatchUsingPure(n_states):
    return [generateMixedStateUsingPure() for n in range(n_states)]
