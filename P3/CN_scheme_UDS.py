# -*- coding: utf-8 -*-
# @Time    :   2021/12/01 23:46:37
# @FileName:   CN_scheme_UDS.py
# @Author  :   ljy
# @Software:   VSCode

# CN隐式格式-一阶迎风

import numpy as np
from numpy.linalg import *
import math
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift


"""
以下是可以修改的地方
count 控制体数量，不是单元
time_step 时间步长
time 时间总长度
"""
count = 50  # count of volume
time_step = 0.001
time = np.arange(0.1, 2.1, time_step)
f = 0.5  # CN格式系数

numerical_T = np.array(np.zeros((count, len(time)+1)))


# [ [[aww aw ae aee ap]], [time] ] +1是为了保存0时刻
a = np.array(np.zeros((count, 3, len(time)+1)))  # 三点差分
S = np.array(np.zeros((count, 2)))  # Sc Sp


def UDS(Gamma, dx, rho, u, Sc, Sp, A, a0, V, L, i, t):  # 每一段控制体进行计算
    D = Gamma / dx
    F = rho * u

    Pe = rho*u/(Gamma*L)

    global a, f
    # a[i, 0] = (D[0] + 1/2 * F[0])*A[0]  # aw
    # a[i, 1] = (D[1] - 1/2 * F[1])*A[1]  # ae
    a[i, 0, t] = D[0] + max(F[0], 0)
    a[i, 1, t] = D[1] + max(-F[1], 0)
    a[i, 2, t] = (a[i, 0, t] + a[i, 1, t] + F[0] *
                  A[0] - F[1]*A[1] - Sp[0]*V[0]) * f + a0  # ap CN格式别忘了这在乘f


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


def matReshape(Ta, Tb, a, a0, dx, S, V, count, t):  # 重组矩阵成为系数矩阵

    # CN格式的f将在此进行处理

    global f
    A = 1  # 截面面积
    b = np.mat(np.zeros((count, 1)))
    # CN格式的Ax=b的b很难处理，目前暂时用循环代替
    for i in range(0, count):
        if i == 0:
            b[i, 0] = a[i, 0] * f * Ta + a[i, 0] * (1-f)*Ta + a[i, 1] * (1-f) * numerical_T[i+1, t-1] + (
                a0 - (1-f)*a[i, 0] - (1-f)*a[i, 1] + (1-f)*S[i, 1]*A*dx[i, 1]) * numerical_T[i, t-1] + S[i, 0] * A * dx[i, 1]
        elif i == count - 1:
            b[i, 0] = a[i, 0]*(1-f) * numerical_T[i-1, t-1] + a[i, 1] * f*Tb + a[i, 1] * (1-f)*Tb + (a0 - (
                1-f)*a[i, 0] - (1-f)*a[i, 1] + (1-f)*S[i, 1]*A*dx[i, 0]) * numerical_T[i, t-1] + S[i, 0] * A * dx[i, 0]
        else:
            b[i, 0] = a[i, 0] * (1-f) * numerical_T[i-1, t-1] + a[i, 1] * (1-f)*numerical_T[i+1, t-1] + (
                a0 - (1-f)*a[i, 0] - (1-f)*a[i, 1] + (1-f)*S[i, 1]*A*dx[i, 1]) * numerical_T[i, t-1] + S[i, 0] * A * dx[i, 1]

    # 对A进行处理
    a[:, 0] = f * a[:, 0]
    a[:, 1] = f * a[:, 1]
    # a的原形式 aw ae ap
    a[:, 0:2] = -1*a[:, 0:2]  # 对aw和ae取负表示移项
    # 交换ap和ae，使得ap在中间，因为matReshape函数假设ap在中间
    a[:, [1, 2]] = a[:, [2, 1]]

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
    """
    以下是可以修改的地方
    Gamma 广义扩散系数，因为一个控制体分为 w e 控制面，因此以1乘2数组表示，下同
    L 一维几何体长度
    crossA 几何体截面积
    u 流速
    rho 密度
    Ta Tb  边界温度
    """

    # 输入参数
    # 广义对流扩散系数 Gamma_w and Gamma_e
    Gamma = np.array([0.1, 0.1], dtype=np.float64)
    L = 1  # 几何尺寸为1
    crossA = np.array([1, 1], dtype=np.float64)  # 截面积恒定  A_w and A_e

    u = np.array([2, 2], dtype=np.float64)  # 风速 u_w and u_e
    # 空气密度千克/立方米 rho_w and rho_e
    rho = np.array([1.0, 1.0], dtype=np.float64)
    Ta = 1
    Tb = 0

    dx = np.array([])
    # 一维问题 V=1
    V = np.array(np.ones((count, 1)))

    global numerical_T

    a0 = rho[0]*(L/count)/time_step
    t_lim = rho[0] * (L/count)**2 / Gamma[0]  # 最大时间步长
    print("时间步长要小于 %f\n" % t_lim)

    Pe = rho * u / (Gamma * L)
    print("Peclet数 Pe = %f" % Pe[0])

    for t in range(1, len(time)):
        t = int(t)
        for i in range(0, count):  # 定时，算区域
            deltaX = distanceOfNodes(L, i+1)
            if i == 0:
                dx = deltaX
            else:
                dx = np.insert(dx, i, values=deltaX, axis=0)
            # V[i] = (dx[i][0]+dx[i][1])/2  # 一维问题V=(dx_W+dx_E)/2
            UDS(Gamma, dx[i], rho, u, S[0], S[1], crossA, a0, V, L, i, t)
        pass

        A, b = matReshape(Ta, Tb, a[:, :, t], a0, dx, S, V, count, t)
        for i in range(count):
            A[i, abs(A[i, :]) < 1e-10] = 0

        numerical = solve(A, b)  # 数值解
        # numerical = np.insert(numerical, 0, values=np.array(np.ones(1)), axis=0)
        numerical_T[:, t] = numerical.flatten()
    pass

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

    # plt.plot(temp_L, analysis)
    # 添加边界点Ta = 1
    Y = np.row_stack((np.ones((1, len(time)+1)), numerical_T))
    # Y = np.insert(numerical_T, 0, values=np.ones((1,len(time)+1)) ,axis=0)
    # 添加边界温度Tb = 0
    Y = np.row_stack((Y, np.zeros((1, len(time)+1))))

    # 添加边界节点位置Xa = 0
    X = np.insert(sumL, 0, values=0)
    # 添加边界节点位置Xb = L
    X = np.insert(X, len(X), values=L)

    '''
    range的步长可以修改
    '''
    for t in range(1, len(time), int(len(time) / 10 + 5)):
        plt.scatter(X, Y[:, t])
    plt.show()

    return A, b


if __name__ == '__main__':
    A, b = fun()
