# 一阶迎风格式UDS
import numpy as np
from numpy.linalg import *
import math
import matplotlib.pyplot as plt

# count of volume
count = 10

a = np.array(np.zeros((count, 3)))  # aw ae ap


def UDS(Gamma, dx, rho, u, Sc, Sp, A, V, L, i):  # 每一段控制体进行计算
    D = Gamma / dx
    F = rho * u

    Pe = rho*u/(Gamma*L)

    global a
    # a[i, 0] = (D[0] + 1/2 * F[0])*A[0]  # aw
    # a[i, 1] = (D[1] - 1/2 * F[1])*A[1]  # ae
    a[i, 0] = D[0] + max(F[0], 0)
    a[i, 1] = D[1] + max(-F[1], 0)
    a[i, 2] = a[i, 0] + a[i, 1] + F[0]*A[0] - F[1]*A[1] - Sp*V  # ap


def distanceOfNodes(L, numOfnode):  # 计算两个节点之间的距离
    dx = np.array(np.zeros((1, 2)), dtype=np.float64)  # 定义类型，否则默认整型
    if numOfnode == 1:  # 第一个节点
        dx[0, 0] = L/(2.0*count)  # 左边节点dx_w = 1/(2*count)
        dx[0, 1] = L/(1.0*count)     # 右边节点dx_e = 1/count

    elif numOfnode == count:  # 最后一个节点
        dx[0, 0] = L/(1.0*count)   # 左边节点
        dx[0, 1] = L/(2.0*count)  # 右边节点
    else:
        dx[0, 0], dx[0, 1] = 1/count, 1/count
    return dx


def matReshape(Ta, Tb, a, count):  # 重组矩阵成为系数矩阵
    b = np.mat(np.zeros((count, 1)))
    b[0, 0] = -a[0, 0]*Ta  # a_w1*Ta
    b[count-1, 0] = -a[count-1, 2]*Tb  # a_en*Tb

    A = np.array(np.zeros([count, count]))
    for i in range(0, count):
        if i == 0:
            A[i, i:i+2] = a[i, i+1:]
        elif i == count-1:
            A[i, i-1:] = a[i, 0:2]
        else:
            A[i, i-1:i+2] = a[i, :]

    return A, b


def fun():
    # 输入参数
    # 广义对流扩散系数 Gamma_w and Gamma_e
    Gamma = np.array([0.1, 0.1], dtype=np.float64)
    L = 1  # 几何尺寸为1
    A = np.array([1, 1], dtype=np.float64)  # 截面积恒定  A_w and A_e
    Sc, Sp = 0, 0  # 无源场
    u = np.array([2, 2], dtype=np.float64)  # 风速 u_w and u_e
    # 空气密度千克/立方米 rho_w and rho_e
    rho = np.array([1.093, 1.093], dtype=np.float64)
    Ta = 50  # 下端温度
    Tb = 20  # 上端温度

    dx = np.array([])
    for i in range(0, count):
        deltaX = distanceOfNodes(L, i+1)
        if i == 0:
            dx = deltaX
        else:
            dx = np.insert(dx, i, values=deltaX, axis=0)
        V = (dx[i][0]+dx[i][1])/2  # 一维问题V=(dx_W+dx_E)/2
        UDS(Gamma, dx[i], rho, u, Sc, Sp, A, V, L, i)

    # A, b = matReshape(Ta, Tb)
    a[:, 0:2] = -1*a[:, 0:2]  # 对aw和ae取负表示移项
    a[:, [1, 2]] = a[:, [2, 1]]  # 交换ap和ae，使得ap在中间，因为matReshape函数假设ap在中间

    A, b = matReshape(Ta, Tb, a, count)
    numerical = solve(A, b)  # 数值解

    dL = dx[:, 0]
    dL[count-1] = dx[count-1, 1]  # 最后一个节点右边距离
    sumL = []
    for i in range(0, count):
        sumL.append(sum(dL[0:i+1]))
    sumL[count-1] = sumL[count-1] + L/(2.0*count)
    sumL = np.array(sumL)

    factor = (Tb-Ta)/(math.exp(rho[0]*u[0]*L/Gamma[0]) - 1)

    temp_L = np.arange(0, 1.01, 0.01)
    analysis = Ta - factor + factor * \
        np.exp(rho[0]*u[0]*temp_L/Gamma[0])  # 解析解

    plt.plot(temp_L, analysis)
    plt.scatter(sumL, numerical.getA())
    plt.show()

    return A, b


if __name__ == '__main__':
    A, b = fun()
