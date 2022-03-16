import numpy as np


def partialTranspose(state):
    state_pt = np.zeros((4, 4)).astype(complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    old_x = i*2+k
                    new_x = i*2+l
                    old_y = j*2+l
                    new_y = j*2+k
                    state_pt[new_y][new_x] = state[old_y][old_x]

    return state_pt


def generalPartialTranspose(state, system):
    state_pt = np.zeros(state.shape).astype(complex)

    for i in range(state.shape[0]):
        for j in range(state.shape[0]):
            i_state = indexToState(i, state)
            j_state = indexToState(j, state)
            i_system = i_state[system]
            j_system = j_state[system]
            i_state[system] = j_system
            j_state[system] = i_system
            i_index = indexFromState(i_state)
            j_index = indexFromState(j_state)
            state_pt[j_index, i_index] = state[j, i]

    return state_pt


def indexToState(index, state):
    state_list = []
    index_length = state.shape[0]
    for qubit in range(int(np.log2(state.shape[0]))):
        if index < int(index_length/2):
            state_list.append(0)
        else:
            state_list.append(1)
            index -= int(index_length/2)
        index_length = int(index_length/2)

    return state_list


def indexFromState(state_list):
    index = 0
    for i in range(len(state_list)):
        index += state_list[i] * 2**(len(state_list) - i - 1)

    return index


state = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

print(generalPartialTranspose(state, 0))
