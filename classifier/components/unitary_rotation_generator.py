import numpy as np
import cmath as math


def randomUnitary(dimension):
    unitary = math.exp(1j*np.random.rand()*2*math.pi)
    for i in range(dimension-1):
        if type(unitary) == np.array:
            unitary = np.matmul(
                unitary, compositeRotationOperator(i+1, dimension))
        else:
            unitary = unitary * compositeRotationOperator(i+1, dimension)
    return unitary


def compositeRotationOperator(n, dimension):
    compositeRotation = 1
    for i in range(n):
        phi = np.random.rand() * math.pi / 2
        psi = np.random.rand() * math.pi * 2
        if n-i == 1:
            chi = np.random.rand() * math.pi * 2
        else:
            chi = 0
        if type(compositeRotation) == np.array:
            compositeRotation = np.matmul(
                compositeRotation, rotationOperator(n-i, n+1, dimension, phi, psi, chi))
        else:
            compositeRotation = compositeRotation * \
                rotationOperator(n-i, n+1, dimension, phi, psi, chi)
    return compositeRotation


def rotationOperator(i, j, n, phi, psi, chi):
    operator = np.zeros((n, n)).astype(complex)

    for k in range(n):
        for l in range(n):
            if k == l:
                if k == i:
                    operator[k, l] = math.exp(1j*psi) * math.cos(phi)
                elif k == j:
                    operator[k, l] = math.exp(-1j*psi) * math.cos(phi)
                else:
                    operator[k, l] = 1
            elif j == k and i == l:
                operator[k, l] = math.exp(1j*chi)*math.sin(phi)
            elif j == l and k == i:
                operator[k, l] = -math.exp(-1j*chi)*math.sin(phi)
    return operator


U = randomUnitary(2**2)
print(np.matmul(U, np.transpose(np.conjugate(U))))
