from .reservoir import Reservoir
from .interface import Interface


class Model:
    def __init__(self):
        pass

    def setSystem(self, system):
        self.system = system

    def setReservoirConnectionFac(self, factory):
        self.reservoirConnectionFac = factory

    def setInterfaceConectionFac(self, factory):
        self.interfaceConnectionFac = factory

    def generateReservoir(self, n_nodes, init_quantum_state=0):
        self.reservoir = Reservoir(self.reservoirConnectionFac)
        self.reservoir.setupNodes(
            n_nodes=n_nodes,
            system_nodes=len(self.system.nodes),
            init_quantum_state=init_quantum_state,
        )
        self.reservoir.setupConnections()

    def generateInterface(self):
        self.interface = Interface(
            sys_nodes=self.system.nodes,
            res_nodes=self.reservoir.nodes.keys(),
            connectionFactory=self.interfaceConnectionFac,
        )
        self.interface.setupConnections()
