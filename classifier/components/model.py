from .reservoir import Reservoir
from .interface import Interface
from .model_log import ModelLog
from .runge_kutta import rungeKutta
from .partial_trace import partialTrace
from .measure_excitations import measureAllExcitations, measureTotalExcitations
from sympy.physics.quantum import TensorProduct
import numpy as np


class Model:
    def __init__(self):
        self.switch_structure_time = 99999

    def setSystem(self, system):
        self.system = system

    def setReservoirInteractionFacs(self, dualFactories, singleFactories):
        self.reservoirDualInteractionFacs = dualFactories
        self.reservoirSingleInteractionFacs = singleFactories

    def setInterfaceInteractionFacs(self, factories):
        self.interfaceInteractionFacs = factories

    def generateReservoir(self, n_nodes, init_quantum_state=0, interaction_rate=1):
        self.reservoir = Reservoir(
            self.reservoirSingleInteractionFacs, self.reservoirDualInteractionFacs)
        self.reservoir.setupNodes(
            n_nodes=n_nodes,
            system_nodes=len(self.system.nodes),
            quantum_state=init_quantum_state,
        )
        self.reservoir.computeInitialQuantumState()
        self.reservoir.setupSingleInteractions()
        self.reservoir.setupDualInteractions(interaction_rate)

    def generateInterface(self, interaction_rate=1):
        self.interface = Interface(
            sys_nodes=self.system.nodes,
            res_nodes=self.reservoir.nodes.keys(),
            interactionFactories=self.interfaceInteractionFacs,
        )
        self.interface.setupInteractions(interaction_rate)

    def setRunDuration(self, run_duration):
        self.run_duration = run_duration

    def setRunResolution(self, run_resolution):
        self.run_timestep = run_resolution

    def setSwitchStructureTime(self, switch_structure_time):
        self.switch_structure_time = switch_structure_time

    def calcIterations(self):
        self.iterations = round(self.run_duration / self.run_timestep)

    def calcStartingState(self):
        self.current_state = TensorProduct(
            self.system.init_quantum_state, self.reservoir.init_quantum_state
        )
        self.calcTraceState()
        self.calcExcitationState()

    def run(self):
        self.structure_phase = 0
        self.setupModelLog()
        self.calcIterations()
        self.calcStartingState()
        self.logIteration()

        for step in range(self.iterations):
            self.updateState()
            self.logIteration()
            if round(self.run_timestep * step, 3) == self.switch_structure_time:
                self.switchStructure()

    def switchStructure(self):
        self.structure_phase = 1

    def logIteration(self):
        self.modelLog.addLogEntry(self.current_state)
        self.modelLog.addTraceLogEntry(self.current_trace_state)
        self.modelLog.addExcitationLogEntry(
            self.current_excitation_expectations)
        self.modelLog.addTotalExcitationLogEntry(
            self.current_total_excitation_expectations
        )
        self.modelLog.moveTimeStep()

    def setupModelLog(self):
        self.modelLog = ModelLog(self.run_timestep)

    def updateState(self):
        self.current_state = rungeKutta(
            self.calcDensityDerivative, self.run_timestep, self.current_state
        )
        self.calcTraceState()
        self.calcExcitationState()

    def calcExcitationState(self):
        self.current_excitation_expectations = measureAllExcitations(
            self.current_trace_state)
        self.current_total_excitation_expectations = measureTotalExcitations(
            self.current_trace_state)

    def calcTraceState(self):
        self.current_trace_state = self.trace(self.current_state, len(
            self.system.nodes), basis=[np.array([[1], [0]]), np.array([[0], [1]])])

    def trace(self, density_matrix, system_nodes, basis):
        for i in range(system_nodes):
            density_matrix = partialTrace(
                density_matrix, trace_system=0, basis=basis)
        return density_matrix

    def calcDensityDerivative(self, state):
        system_component = self.system.calcDensityDerivative(
            state, self.structure_phase)
        reservoir_component = self.reservoir.calcDensityDerivative(
            state, self.structure_phase)
        interface_component = self.interface.calcDensityDerivative(
            state, self.structure_phase)
        return reservoir_component + system_component + interface_component
