import numpy as np
import matplotlib.pyplot as plt

# 域离散
L = 0.012      # 棒总长
N = 24        # 控制单元数目
dx = L / N    # 控制单元长度
centriod = [i * dx + dx / 2 for i in range(N)]   # 控制单元中心坐标
S = [1] * N                                      # 控制单元中心对应的界面面积
k = np.zeros(N)                               # 控制体单元中心处的导热系数
for i in range(2, N-2):
    k[i] = 25

k[0] = 200
k[1] = 200
k[N-1] = 200
k[N-2] = 200
T = [0] * N                                      # 控制体单元中心温度初始值。可以任意给定,这个没看懂哦
A = np.zeros((N, N))                             # 整体系数矩阵
b = np.zeros(N)                                  # 整体方程右端项
q = np.zeros(N)                                 # 控制单元上的源值
# 控制体单元中心处的导热系数
for i in range(2, N-2):
    q[i] = 100000000

q[0] = 0
q[1] = 0
q[N-1] = 0
q[N-2] = 0
TA = 100
TB = 100

for ii in range(1, N - 1):  # 用来组装整体系数矩阵的整体循环，注意边界单元没有循环
    # C对应下标ii, W对应下标ii-1, E对应下标ii+1
    a_W = -((k[ii - 1] + k[ii])*S[ii]) / 2 / dx
    a_E = -((k[ii] + k[ii + 1])*S[ii]) / 2 / dx
    S_C = q[ii] * S[ii] * dx                               # 源项
    A[ii, ii - 1] = a_W
    A[ii, ii + 1] = a_E
    A[ii, ii] = -(a_W + a_E)
    b[ii] = S_C

# 第一个边界单元修正， 此时C对应下标0， k_A = k[0]
A[0, 1] = -(k[0] + k[1]) / 2 / dx
b[0] = 2 * k[0] / dx * TA + q[0] * dx
A[0, 0] = -A[0, 1] + 2 * k[0] / dx
A[N-1, N-2] = -(k[N-2] + k[N-1]) / 2 / dx
b[N-1] = k[N-1] * 2 / dx * TB + q[N-1] * dx
A[N-1, N-1] = -A[N-1, N-2] + k[N-1] * 2 / dx
T = np.linalg.solve(A, b)

fig = plt.figure()
plt.plot(np.arange(N) / N * L, T, 'o')
plt.show()

x = np.array([0, *centriod, L])
exact = np.zeros(26)

T_all = np.array([TA, *T, TB])
for i in range(3, 22):
    m = ((T[N-3] - T[2]) / L + q[0] / 2 / k[0] * (x[22] - x[i])) * (x[i] - x[2]) + T[2]
    exact[i] = m

for i in range(3):
    exact[i] = ((T[2] - TA) / T[2]) * x[i] + TA

for i in range(23, 25):
    exact[i] = ((TB - T[23]) / (T[23] - T[21])) * (x[i] - x[23]) + T[23]

print(T)
fig = plt.figure()
plt.plot(x, T_all, 'o', label='FVM')
plt.plot(x, exact, '-', label='Exact')
plt.xlabel("燃料棒直径")
plt.ylabel("温度")
plt.legend()
plt.show()
