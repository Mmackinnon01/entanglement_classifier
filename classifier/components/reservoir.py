from node import Node


class Reservoir:
    
    def __init__(self):
        self.nodes = {}

    def setupNodes(self, n_nodes, quantum_state = 0):
        for n in range(n_nodes):
            self.nodes[n] = Node(quantum_state)
