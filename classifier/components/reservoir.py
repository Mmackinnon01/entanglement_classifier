from .node import Node
from sympy.physics.quantum import TensorProduct
from random import shuffle


class Reservoir:
    def __init__(self, singleInteractionFactories, dualInteractionFactories):
        self.nodes = {}
        self.dualInteractions = {}
        self.singleInteractions = {}
        self.singleInteractionFactories = singleInteractionFactories
        self.dualInteractionFactories = dualInteractionFactories

    def setupNodes(self, n_nodes, system_nodes, quantum_state=0):
        self.system_nodes = system_nodes
        for n in range(n_nodes):
            self.nodes[system_nodes + n] = Node(
                node_id=system_nodes + n, init_quantum_state=quantum_state
            )

    def remove_interactions(self, interactions, interaction_rate):
        shuffle(interactions)
        count = int(len(interactions) * interaction_rate)
        if not count:
            return []  # edge case, no elements removed
        return interactions[:count]

    def setupDualInteractions(self, interaction_rate=1):
        node_id_list = self.nodes.keys()
        node_pairs = [[x, y]
                      for x in node_id_list for y in node_id_list if x < y]

        node_pairs = self.remove_interactions(node_pairs, interaction_rate)

        for node_pair in node_pairs:
            self.setupIndividualDualInteraction(
                node1=node_pair[0], node2=node_pair[1])

    def setupIndividualDualInteraction(self, node1, node2):
        for i, factory in enumerate(self.dualInteractionFactories):
            if type(factory) == list:
                interaction_list = []
                for fac in factory:
                    interactino = fac.generateInteraction(
                        [node1, node2], n_nodes=len(
                            self.sys_nodes) + len(self.res_nodes)
                    )
                    interaction_list.append(interactino)
                self.dualInteractions["res_interaction_{}_{}{}".format(
                    i, node1, node2)] = interaction_list
            else:
                interaction = factory.generateInteraction(
                    [node1, node2], n_nodes=self.system_nodes + len(self.nodes)
                )
                self.dualInteractions["res_interaction_{}_{}{}".format(
                    i, node1, node2)] = interaction

    def setupSingleInteractions(self):
        node_id_list = self.nodes.keys()
        for node in node_id_list:
            self.setupIndividualSingleInteraction(
                node=node)

    def setupIndividualSingleInteraction(self, node):
        for i, factory in enumerate(self.singleInteractionFactories):
            if type(factory) == list:
                interaction_list = []
                for fac in factory:
                    interactino = fac.generateInteraction(
                        node, n_nodes=len(
                            self.sys_nodes) + len(self.res_nodes)
                    )
                    interaction_list.append(interactino)
                self.singleInteractions["res_interaction_{}_{}".format(
                    i, node)] = interaction_list
            else:
                interaction = factory.generateInteraction(
                    node, n_nodes=self.system_nodes + len(self.nodes)
                )
                self.singleInteractions["res_interaction_{}_{}".format(
                    i, node)] = interaction

    def computeInitialQuantumState(self):
        total_state = 0
        for node in self.nodes.values():
            if type(total_state) == int:
                total_state = node.init_quantum_state
            else:
                total_state = TensorProduct(
                    total_state, node.init_quantum_state)
        self.init_quantum_state = total_state

    def calcDensityDerivative(self, model_state, structure_phase):
        density_derivative = 0
        for interaction in self.dualInteractions.values():
            if type(interaction) == list:
                interaction = interaction[structure_phase]
            density_derivative += interaction.calc(model_state)

        for interaction in self.singleInteractions.values():
            if type(interaction) == list:
                interaction = interaction[structure_phase]
            density_derivative += interaction.calc(model_state)
        return density_derivative
