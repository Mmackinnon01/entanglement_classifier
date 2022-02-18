import numpy as np


class Interaction:
    def __init__(self, nodes, function):
        self.nodes = nodes
        self.function = function

    def calc(self, model_state):
        return self.function.calc(model_state)


class InteractionFactory:
    def __init__(self, template_function, **variable_values):
        self.template_function = template_function
        self.variable_values = variable_values

    def generateSpecificFunction(self, nodes, n_nodes):
        kwargs = self.generateRandomArguments()

        return self.template_function(nodes, n_nodes, **kwargs)

    def generateRandomArguments(self):
        kwargs = self.variable_values.copy()
        for kwarg, value in kwargs.items():
            if type(value) == list:
                lower_bound = value[0]
                upper_bound = value[1]
                kwargs[kwarg] = np.random.uniform(lower_bound, upper_bound)
        return kwargs

    def generateInteraction(self, nodes, n_nodes):
        interaction_function = self.generateSpecificFunction(
            nodes, n_nodes)
        return Interaction(nodes, interaction_function)
