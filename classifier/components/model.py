from .reservoir import Reservoir
from .interface import Interface
from .model_log import ModelLog
from .runge_kutta import rungeKutta
from sympy.physics.quantum import TensorProduct


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
            quantum_state=init_quantum_state,
        )
        self.reservoir.computeInitialQuantumState()
        self.reservoir.setupConnections()

    def generateInterface(self):
        self.interface = Interface(
            sys_nodes=self.system.nodes,
            res_nodes=self.reservoir.nodes.keys(),
            connectionFactory=self.interfaceConnectionFac,
        )
        self.interface.setupConnections()

    def setRunDuration(self, run_duration):
        self.run_duration = run_duration

    def setRunResolution(self, run_resolution):
        self.run_timestep = run_resolution

    def calcIterations(self):
        self.iterations = round(self.run_duration / self.run_timestep)

    def calcStartingState(self):
        self.current_state = TensorProduct(
            self.system.init_quantum_state, self.reservoir.init_quantum_state
        )
        self.modelLog.addLogEntry(self.current_state)

    def run(self):
        self.setupModelLog()
        self.calcIterations()
        self.calcStartingState()

        for step in range(self.iterations):
            self.updateState()
            self.logIteration()

    def logIteration(self):
        self.modelLog.addLogEntry(self.current_state)

    def setupModelLog(self):
        self.modelLog = ModelLog(self.run_timestep)

    def updateState(self):
        self.current_state = rungeKutta(
            self.calcDensityDerivative, self.run_timestep, self.current_state
        )

    def calcDensityDerivative(self, state):
        system_component = self.system.calcDensityDerivative(state)
        reservoir_component = self.reservoir.calcDensityDerivative(state)
        interface_component = self.interface.calcDensityDerivative(state)
        return reservoir_component + system_component + interface_component
