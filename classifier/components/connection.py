import numpy as np


class Connection:
    def __init__(self, node1, node2, function):
        self.node1 = node1
        self.node2 = node2
        self.function = function

    def calc(self, model_state):
        return self.function.calc(model_state)


class HamiltonianConnection:
    def __init__(self, node1, node2, function):
        self.node1 = node1
        self.node2 = node2
        self.function = function

    def commutator(self, a, b):
        return np.matmul(a, b) - np.matmul(b, a)

    def calc(self, model_state):
        return -1j * self.commutator(self.function.calc(model_state), model_state)


class ConnectionFactory:
    def __init__(self, template_function, is_hamiltonian=False, **variable_values):
        self.template_function = template_function
        self.variable_values = variable_values
        self.is_hamiltonian = is_hamiltonian

    def generateSpecificFunction(self, node1, node2, n_nodes):
        kwargs = self.generateRandomArguments()

        return self.template_function(node1, node2, n_nodes, **kwargs)

    def generateRandomArguments(self):
        kwargs = self.variable_values.copy()
        for kwarg, value in kwargs.items():
            if type(value) == list:
                lower_bound = value[0]
                upper_bound = value[1]
                kwargs[kwarg] = np.random.uniform(lower_bound, upper_bound)
        return kwargs

    def generateConnection(self, node1, node2, n_nodes):
        connection_function = self.generateSpecificFunction(
            node1, node2, n_nodes)
        if self.is_hamiltonian:
            return HamiltonianConnection(node1, node2, connection_function)
        else:
            return Connection(node1, node2, connection_function)
