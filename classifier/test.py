import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sympy.physics.quantum import TensorProduct

from components.system import System
from components.interaction import InteractionFactory, Interaction
from components.model import Model
from components.interaction_functions import CascadeFunction, EnergyExchangeFunction, DampingFunction
from components.state_generator import generateMixedBatch, generatePureBatch

import concurrent
import copy

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

samples = 200000
test_size = 0.5
x_pure = generatePureBatch(int(samples/2), dim=4)
y_pure = np.ones(int(samples/2))
x_mixed = generateMixedBatch(int(samples/2), dim=4)
y_mixed = np.zeros(int(samples/2))
x = np.concatenate([x_pure, x_mixed])
y = np.concatenate([y_pure, y_mixed])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=42, shuffle=True)
reservoir_nodes = 2
system_nodes = 2


def transform(state, label):
    system_state = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    system_node_list = [0, 1]

    if len(system_node_list) != system_nodes:
        raise Exception

    system_interactions = {"sys_interaction_0": Interaction(
        0, DampingFunction(0, reservoir_nodes+system_nodes, 1)),
        "sys_interaction_1": Interaction(
        1, DampingFunction(1, reservoir_nodes+system_nodes, 1))}

    system = System(
        init_quantum_state=system_state, nodes=system_node_list, interactions=system_interactions
    )
    interfaceFactory1 = InteractionFactory(
        CascadeFunction, gamma_1=1, gamma_2=1)
    interfaceFactory2 = InteractionFactory(
        CascadeFunction, gamma_1=0, gamma_2=0)
    reservoirFactory1 = InteractionFactory(
        EnergyExchangeFunction, coupling_strength=1)
    reservoirFactory2 = InteractionFactory(
        DampingFunction, damping_strength=1)
    model = Model()
    model.setSystem(system)
    model.setReservoirInteractionFacs(
        dualFactories=[reservoirFactory1], singleFactories=[reservoirFactory2])
    model.setInterfaceInteractionFacs([[interfaceFactory1, interfaceFactory2]])
    model.generateReservoir(
        reservoir_nodes, init_quantum_state=0, interaction_rate=1)
    model.generateInterface(interaction_rate=.5)
    model.setRunDuration(1.8)
    model.setRunResolution(0.01)
    model.setSwitchStructureTime(1)
    return [model.transform(state), label]


d_train = []
d_test = []

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        progress = 0
        for i, state in enumerate(x_train):
            futures.append(executor.submit(
                transform, state=state, label=y_train[i]))
        for fut in concurrent.futures.as_completed(futures):
            progress += 1
            print(progress)
            d_train.append(fut.result())

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, state in enumerate(x_test):
            futures.append(executor.submit(
                transform, state=state, label=y_test[i]))
        for fut in concurrent.futures.as_completed(futures):
            progress += 1
            print(progress)
            d_test.append(fut.result())

    x_train = [d[0] for d in d_train]
    y_train = [d[1] for d in d_train]
    x_test = [d[0] for d in d_test]
    y_test = [d[1] for d in d_test]

    mlp = MLPClassifier(max_iter=20000).fit(
        np.array(x_train), y_train)
    print(mlp.score(np.array(x_test), y_test))
