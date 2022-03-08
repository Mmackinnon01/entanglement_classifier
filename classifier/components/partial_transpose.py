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


state = [[1j, 2j, 3j, 4j],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
