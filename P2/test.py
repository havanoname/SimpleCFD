from scipy.sparse import spdiags
import numpy as np

a = np.array([[30, 19, 40, 22, 33], [30, 19, 30, 33, 22], [20, 19, 30, 23, 32]])


def ShiftRows(x, row):
    A = np.array(np.zeros([row, row]))
    for i in range(0, row):
        if i == 0:
            A[i][i:i+2] = a[i][i+1:]
        elif i == row-1:
            A[i][i-1:] = a[i][i-1:]
        else:
            A[i][i-1:i+2] = a[i][:]

    return A


A = ShiftRows(a, 3)

