from .node import Node


class Reservoir:
    def __init__(self, connectionFactory):
        self.nodes = {}
        self.connections = {}
        self.connectionFactory = connectionFactory

    def setupNodes(self, n_nodes, system_nodes, quantum_state=0):
        for n in range(n_nodes):
            self.nodes[system_nodes + n] = Node(
                node_id=system_nodes + n, init_quantum_state=quantum_state
            )

    def setupConnections(self):
        node_id_list = self.nodes.keys()
        node_pairs = [[x, y] for x in node_id_list for y in node_id_list if x < y]

        for node_pair in node_pairs:
            self.setupIndividualConnection(node1=node_pair[0], node2=node_pair[1])

    def setupIndividualConnection(self, node1, node2):
        connection = self.connectionFactory.generateConnection(node1, node2)
        self.connections["res_connection_{}{}".format(node1, node2)] = connection
