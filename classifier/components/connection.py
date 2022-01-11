import numpy as np


class Connection:
    def __init__(self, node1, node2, function):
        self.node1 = node1
        self.node2 = node2
        self.function = function

    def calc(self, **kwargs):
        return self.function(**kwargs)


class ConnectionFactory:
    def __init__(self, template_function, **variable_values):
        self.template_function = template_function
        self.variable_values = variable_values

    def generateSpecificFunction(self):
        kwargs = self.generateRandomArguments()

        def specific_function(**input_kwargs):
            for kwarg, value in input_kwargs.items():
                kwargs[kwarg] = value
            return self.template_function(**kwargs)

        return specific_function

    def generateRandomArguments(self):
        kwargs = self.variable_values.copy()
        for kwarg, value in kwargs.items():
            if type(value) == list:
                lower_bound = value[0]
                upper_bound = value[1]
                kwargs[kwarg] = np.random.uniform(lower_bound, upper_bound)

        return kwargs

    def generateConnection(self, node1, node2):
        connection_function = self.generateSpecificFunction()
        return Connection(node1, node2, connection_function)
