def rungeKutta(f, h, state):
    k1 = runge_kutta_k1(f, h, state)
    k2 = runge_kutta_k2(f, h, state, k1)
    k3 = runge_kutta_k3(f, h, state, k2)
    k4 = runge_kutta_k4(f, h, state, k3)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def runge_kutta_k1(f, h, state):
    return f(state)


def runge_kutta_k2(f, h, state, k1):
    return f(state + h * (k1 / 2))


def runge_kutta_k3(f, h, state, k2):
    return f(state + h * (k2 / 2))


def runge_kutta_k4(f, h, state, k3):
    return f(state + h * k3)
