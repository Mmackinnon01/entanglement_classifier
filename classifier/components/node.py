import numpy as np


spin_up = np.array([[1, 0], [0, 0]])
spin_down = np.array([[0, 0], [0, 1]])


class Node:
    def __init__(self, node_id, init_quantum_state=0):
        self.node_id = node_id
        self.init_quantum_state = init_quantum_state

    @property
    def init_quantum_state(self):
        return self._init_quantum_state

    @init_quantum_state.setter
    def init_quantum_state(self, quantum_state: int):
        if quantum_state == "random":
            quantum_state = round(np.random.rand())

        if quantum_state == 0:
            self._init_quantum_state = spin_down
        elif quantum_state == 1:
            self._init_quantum_state = spin_up
        else:
            raise ValueError("Invalid quantum state provided")
