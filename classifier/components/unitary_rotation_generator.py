import numpy as np
import cmath as math


def randomUnitary(dimension):
    unitary = math.exp(1j*np.random.rand()*2*math.pi)
    first = True
    for i in range(dimension-1):
        i += 1
        if first:
            unitary = unitary * compositeRotationOperator(i, dimension)
            first = False
        else:
            unitary = np.matmul(
                unitary, compositeRotationOperator(i, dimension))

    return unitary


def compositeRotationOperator(n, dimension):
    compositeRotation = 1
    first = True
    for i in range(n):
        i += 1
        phi = np.random.rand() * math.pi / 2
        psi = np.random.rand() * math.pi * 2
        if n == i:
            chi = np.random.rand() * math.pi * 2
        else:
            chi = 0
        if first:
            compositeRotation = rotationOperator(
                n-i+1, n+1, dimension, phi, psi, chi)
            first = False
        else:
            compositeRotation = np.matmul(
                compositeRotation, rotationOperator(n-i+1, n+1, dimension, phi, psi, chi))

    return compositeRotation


def rotationOperator(i, j, n, phi, psi, chi):
    operator = np.zeros((n, n)).astype(complex)
    i -= 1
    j -= 1
    for k in range(n):
        for l in range(n):
            if k == l:
                if k == i:
                    operator[k, l] = math.exp(1j*psi) * math.cos(phi)
                elif k == j:
                    operator[k, l] = math.exp(-1j*psi) * math.cos(phi)
                else:
                    operator[k, l] = 1
            elif i == k and j == l:
                operator[k, l] = math.exp(1j*chi)*math.sin(phi)
            elif i == l and j == k:
                operator[k, l] = -math.exp(-1j*chi)*math.sin(phi)
    return operator


U = randomUnitary(8)
U_dagger = np.transpose(np.conjugate(U))

print(np.round(np.matmul(U, U_dagger), 2))
