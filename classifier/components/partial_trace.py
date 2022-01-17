import numpy as np
from sympy.physics.quantum import TensorProduct


def ket(bra):
    return np.conjugate(np.transpose(bra))


identity = np.array([[1, 0],
                    [0, 1]])


def partialTrace(state, trace_system, basis):
    sub_state = 0
    num_systems = np.log2(state.shape[0])
    for basis_vector in basis:
        left_term, right_term = calcTraceSumTerms(
            trace_system, num_systems, basis_vector)
        sub_state += np.linalg.multi_dot([left_term, state, right_term])
    return sub_state


def calcTraceSumTerms(trace_system, num_systems, basis_vector):
    left_term = 0
    right_term = 0
    for system in np.arange(num_systems):
        if system == trace_system:
            left_term = updateTraceTerm(ket(basis_vector), left_term)
            right_term = updateTraceTerm(basis_vector, right_term)
        else:
            left_term = updateTraceTerm(identity, left_term)
            right_term = updateTraceTerm(identity, right_term)
    return left_term, right_term


def updateTraceTerm(system_component, term):
    if type(term) != np.ndarray:
        term = system_component
    else:
        term = TensorProduct(term, system_component)
    return term
