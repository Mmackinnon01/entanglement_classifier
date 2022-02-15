class System:
    def __init__(self, init_quantum_state, nodes: list, connections: dict):
        self.init_quantum_state = init_quantum_state
        self.nodes = nodes
        self.connections = connections

    def calcDensityDerivative(self, model_state, structure_phase):
        density_derivative = 0
        for connection in self.connections.values():
            if type(connection) == list:
                connection = connection[structure_phase]
            density_derivative += connection.calc(model_state)
        return density_derivative
