class Interface:
    def __init__(self, sys_nodes: list, res_nodes: list, connectionFactory):
        self.sys_nodes = sys_nodes
        self.res_nodes = res_nodes
        self.connections = {}
        self.connectionFactory = connectionFactory

    def setupConnections(self):
        node_pairs = [
            [sys_node, res_node]
            for sys_node in self.sys_nodes
            for res_node in self.res_nodes
        ]

        for node_pair in node_pairs:
            self.setupIndividualConnection(node1=node_pair[0], node2=node_pair[1])

    def setupIndividualConnection(self, node1, node2):
        connection = self.connectionFactory.generateConnection(
            node1, node2, n_nodes=len(self.sys_nodes) + len(self.res_nodes)
        )
        self.connections["interface_connection_{}{}".format(node1, node2)] = connection

    def calcDensityDerivative(self, model_state):
        density_derivative = 0
        for connection in self.connections.values():
            density_derivative += connection.calc(model_state)
        return density_derivative
