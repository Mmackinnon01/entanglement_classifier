import numpy as np
import math
from sympy.physics.quantum import TensorProduct
from components.rotation import generateBipartiteRotation


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


def generateSeparableState(dim=4):
    state = 0
    for i in range(int(math.log(dim, 2))):
        new_state = generateMixedState()
        if type(state) == int:
            state = new_state
        else:
            state = TensorProduct(state, new_state)
    return state


def generateEntangledState(dim=4):
    a = 1 - np.random.rand()/2
    bell_t = np.array([1/(2**0.5), 0, 0, 1/(2**0.5)])
    bell = np.array([[1/(2**0.5)], [0], [0], [1/(2**0.5)]])
    bell_ro = TensorProduct(bell, bell_t)
    mat = a*bell_ro + (1-a)*generateMixedState(dim)
    rotation = generateBipartiteRotation()
    mat = np.linalg.multi_dot(
        [rotation, mat, np.transpose(np.conjugate(rotation))])
    return mat


def generatePureBatch(n_states, dim=2):
    return [generatePureState(dim) for n in range(n_states)]


def generateMixedBatch(n_states, dim=2):
    return [generateMixedState(dim) for n in range(n_states)]


def generateEntangledBatch(n_states, dim=4):
    return [generateEntangledState(dim) for n in range(n_states)]


def generateSeparableBatch(n_states, dim=4):
    return [generateSeparableState(dim) for n in range(n_states)]
