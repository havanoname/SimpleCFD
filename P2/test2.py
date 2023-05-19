import numpy as np
from scipy.ndimage.interpolation import shift

a = np.array(
    [[30, 19, 40, 22, 33], [30, 19, 30, 33, 22],  [30, 19, 30, 33, 22], [30, 19, 30, 33, 22], [20, 19, 30, 23, 32]])

# 这玩意比scipy快，但是我们对速度没啥要求
# def shift(xs, n):
#     if n >= 0:
#         return np.concatenate((np.full(n, np.nan), xs[:-n]))
#     else:
#         return np.concatenate((xs[-n:], np.full(-n, np.nan)))
row = 5


temp = np.array(np.ones((10, 2)))
mul = np.array(np.zeros((10, 1)))
# b = temp[:, 0]
print(temp[:, 0] * mul[:, 0])
print(np.shape(temp[:, 0]))
print(np.shape(mul[:, 0]))
# print(temp * mul)
# print(b * mul.T)

# mul1 = np.array([[1, 2, 3, 4, 5]])
# print(mul1)
# print(mul1.T)
# print(a[:, 0] * mul1)
# print(a[:, 0])


def ShiftRows(x, row):
    for i in range(0, row):
        x[i][:] = shift(x[i][:], i-1, cval=0)
    return x


# 扩充矩阵并且补0
b = np.zeros((9, 9))
b[0:row, 0:row] = a

# a[1,a[1,:] > 1] = 1

A = ShiftRows(a, row)
# print(A)
