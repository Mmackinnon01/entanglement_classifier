from random import shuffle


class Interface:
    def __init__(self, sys_nodes: list, res_nodes: list, interactionFactories):
        self.sys_nodes = sys_nodes
        self.res_nodes = res_nodes
        self.singleInteractions = {}
        self.interactions = {}
        self.interactionFactories = interactionFactories

    def remove_interactions(self, interactions, interaction_rate):
        shuffle(interactions)
        count = int(len(interactions) * interaction_rate)
        if not count:
            return []  # edge case, no elements removed
        return interactions[:count]

    def setupInteractions(self, interaction_rate=1):
        node_pairs = [
            [sys_node, res_node]
            for sys_node in self.sys_nodes
            for res_node in self.res_nodes
        ]

        node_pairs = self.remove_interactions(node_pairs, interaction_rate)

        for node_pair in node_pairs:
            self.setupIndividualInteraction(
                node1=node_pair[0], node2=node_pair[1])

    def setupIndividualInteraction(self, node1, node2):
        for i, factory in enumerate(self.interactionFactories):
            if type(factory) == list:
                interaction_list = []
                for fac in factory:
                    interaction = fac.generateInteraction(
                        [node1, node2], n_nodes=len(
                            self.sys_nodes) + len(self.res_nodes)
                    )
                    interaction_list.append(interaction)
                self.interactions["interface_interaction_{}_{}{}".format(
                    i, node1, node2)] = interaction_list
            else:
                interaction = factory.generateInteraction(
                    [node1, node2], n_nodes=len(
                        self.sys_nodes) + len(self.res_nodes)
                )
                self.interactions["interface_interaction_{}_{}{}".format(
                    i, node1, node2)] = interaction

    def calcDensityDerivative(self, model_state, structure_phase):
        density_derivative = 0
        for interaction in self.interactions.values():
            if type(interaction) == list:
                interaction = interaction[structure_phase]
            density_derivative += interaction.calc(model_state)
        return density_derivative
