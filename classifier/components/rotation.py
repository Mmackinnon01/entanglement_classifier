import math
import cmath
import numpy as np
from sympy import tensorproduct
from sympy.physics.quantum import TensorProduct


def rotationX(theta):
    return np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
                     [-1j*math.sin(theta/2), math.cos(theta/2)]])


def rotationY(theta):
    return np.array([[math.cos(theta/2), -math.sin(theta/2)],
                     [math.sin(theta/2), math.cos(theta/2)]])


def rotationZ(theta):
    return np.array([[cmath.exp(-1j*theta/2), 0],
                     [0, cmath.exp(1j*theta/2)]])


def generateRandomRotation():
    x = np.random.rand() * 2 * math.pi
    y = np.random.rand() * 2 * math.pi
    z = np.random.rand() * 2 * math.pi
    rotation = np.linalg.multi_dot([rotationX(x), rotationY(y), rotationZ(z)])
    return rotation


def generateBipartiteRotation():
    return TensorProduct(generateRandomRotation(), generateRandomRotation())
