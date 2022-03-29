import numpy as np

from components.partial_trace import partialTrace
from components.partial_transpose import partialTranspose


def assessEntanglement(state):
    dim = int(np.log2(state.shape[0]))
    for qubit in range(dim):
        mat = partialTranspose(state, qubit)
        min_eig = min(np.linalg.eigvals(mat))
        if min_eig < 0:
            return dim-1

    if dim > 2:
        for qubit in range(dim):
            entangled = assessEntanglement(partialTrace(
                state, qubit, basis=[np.array([[1], [0]]), np.array([[0], [1]])]))
            if entangled != 0:
                return entangled

    return 0
