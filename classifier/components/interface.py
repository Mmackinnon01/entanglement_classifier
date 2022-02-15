from random import shuffle


class Interface:
    def __init__(self, sys_nodes: list, res_nodes: list, connectionFactory):
        self.sys_nodes = sys_nodes
        self.res_nodes = res_nodes
        self.connections = {}
        self.connectionFactory = connectionFactory

    def remove_connections(self, connections, connection_rate):
        shuffle(connections)
        count = int(len(connections) * connection_rate)
        if not count:
            return []  # edge case, no elements removed
        return connections[:count]

    def setupConnections(self, connection_rate=1):
        node_pairs = [
            [sys_node, res_node]
            for sys_node in self.sys_nodes
            for res_node in self.res_nodes
        ]

        node_pairs = self.remove_connections(node_pairs, connection_rate)

        for node_pair in node_pairs:
            self.setupIndividualConnection(
                node1=node_pair[0], node2=node_pair[1])

    def setupIndividualConnection(self, node1, node2):
        if type(self.connectionFactory) == list:
            connection_list = []
            for factory in self.connectionFactory:
                connection = factory.generateConnection(
                    node1, node2, n_nodes=len(
                        self.sys_nodes) + len(self.res_nodes)
                )
                connection_list.append(connection)
            self.connections["interface_connection_{}{}".format(
                node1, node2)] = connection_list
        else:
            connection = self.connectionFactory.generateConnection(
                node1, node2, n_nodes=len(self.sys_nodes) + len(self.res_nodes)
            )
            self.connections["interface_connection_{}{}".format(
                node1, node2)] = connection

    def calcDensityDerivative(self, model_state, structure_phase):
        density_derivative = 0
        for connection in self.connections.values():
            if type(connection) == list:
                connection = connection[structure_phase]
            density_derivative += connection.calc(model_state)
        return density_derivative
