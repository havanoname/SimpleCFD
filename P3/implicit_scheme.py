# 隐式格式-QUICK

import numpy as np
from numpy.linalg import *
import math
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift


# count of volume
count = 5
time_step = 0.1
time = np.arange(0.1, 2.1, time_step)
f = 0.5  # CN格式系数

numerical_T = np.array(np.zeros((count, len(time)+1)))
# [ [[aww aw ae aee ap]], [time] ] +1是为了保存0时刻
a = np.array(np.zeros((count, 5, len(time)+1)))  # 五点差分
S = np.array(np.zeros((count, 2)))  # Sc Sp


def IMS(Gamma, dx, rho, u, A, a0, V, L, i, t):  # 每一段控制体进行计算
    D = Gamma / dx
    F = rho * u

    t = int(t)  # 转换为整型
    alpha = 0
    if u[0] > 0:
        alpha = 1
    if u[1] > 0:
        alpha = 1

    Pe = rho*u/(Gamma*L)

    global a, S, f

    # a[i, 0] = (D[0] + 1/2 * F[0])*A[0]  # aw
    # a[i, 1] = (D[1] - 1/2 * F[1])*A[1]  # ae
    a[i, 0, t] = -1/8 * alpha * F[0]  # aww
    a[i, 1, t] = D[0] + 6/8 * alpha * F[0] + 1/8 * \
        alpha * F[1] + 3/8 * (1-alpha)*F[0]  # aw
    a[i, 2, t] = D[1] - 3/8 * alpha*F[1] - 6/8 * (1-alpha) * F[1] - 1 / \
        8 * (1-alpha) * F[0]   # ae
    a[i, 3, t] = 1/8 * (1-alpha) * F[1]  # aee
    # 使用QUICK格式在边界点的处理
    Da_start = Gamma[0]/dx[1]  # 在第一个节点才会用，所以dx[1] = 0.1
    Db_start = Gamma[1]/dx[0]  # 在最后一个节点才会用，所以dx[0] = 0.1
    if i == 0:  # 第一个节点
        a[i, 0, t] = 0   # aww = 0
        a[i, 1, t] = 0  # aw = 0
        # ae = De + 1/3D_{A*} - 3/8Fe.  D_{A*} = gamma/dx[1] PS 是dx[1]不是0
        a[i, 2, t] = D[1] + 1/3*Da_start - 3/8*F[1]
        a[i, 3, t] = 0  # aee = 0
        S[i, 0] = 8/3 * Da_start + 2/8*F[1] + F[0]  # Sc
        S[i, 1] = -1 * (8/3 * Da_start + 2/8 * F[1] + F[0])  # Sp
    elif i == 1:
        a[i, 0, t] = 0  # aww
        a[i, 1, t] = D[0]+7/8*F[0]+1/8*F[1]  # aw
        a[i, 2, t] = D[1] - 3/8*F[1]  # ae
        a[i, 3, t] = 0  # aee
        S[i, 0] = -1/4 * F[0]
        S[i, 1] = 1/4 * F[0]
    elif i == count - 1:  # 最后一个节点
        a[i, 0, t] = -1/8*F[0]  # aww
        a[i, 1, t] = D[0] + 1/3*Db_start + 6/8*F[0]  # aw
        a[i, 2, t] = 0  # ae
        a[i, 3, t] = 0     # aee = 0
        S[i, 0] = (8/3 * Db_start - F[1])  # Sc
        S[i, 1] = -(8/3 * Db_start - F[1])  # Sp

    # 前面判断完再计算ap
    a[i, 4, t] = a0 + (a[i, 0, t] + a[i, 1, t] + a[i, 2, t] + \
        a[i, 3, t] + F[0]*A[0] - F[1]*A[1]) * f - S[i, 1] * V[i]  # ap


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

    # 注意在该函数被调用前，a已经被更改为 aww aw ap ae aee

    global f
    A = 1  # 截面面积
    b = np.mat(np.zeros((count, 1)))
    # CN格式的Ax=b的b很难处理，目前暂时用循环代替
    for i in range(0, count):
        if i == 0:
            b[i, 0] = a[i, 0] * f * Ta + a[i, 0] * (1-f)*Ta + a[i, 1] * (1-f) * numerical_T[i+1, t-1] + (
                a0 - (1-f)*a[i, 0] - (1-f)*a[i, 1] + (1-f)*S[i,1]*A*dx[i,1]) * numerical_T[i, t-1] + S[i,0] * A * dx[i,1]
        elif i == count - 1:
            b[i, 0] = a[i, 0]*(1-f) * numerical_T[i-1, t-1] + a[i, 1] * f*Tb + a[i, 1] * (1-f)*Tb + (a0 - (
                1-f)*a[i, 0] - (1-f)*a[i, 1] + (1-f)*S[i,1]*A*dx[i,0]) * numerical_T[i, t-1] + S[i,0] * A * dx[i,0]
        else:
            b[i, 0] = a[i, 0] * (1-f) * numerical_T[i-1, t-1] + a[i, 1] * (1-f)*numerical_T[i+1, t-1] + (
                a0 - (1-f)*a[i, 0] - (1-f)*a[i, 1] + (1-f)*S[i,1]*A*dx[i,1]) * numerical_T[i, t-1] + S[i,0] * A * dx[i,1]

        # a的原形式 aww aw ae aee ap
    a[:, 0:3, t] = -1*a[:, 0:3, t]  # 对aw和ae取负表示移项
    # 交换ap和ae，使得ap在中间，因为matReshape函数假设ap在中间 aww aw ap aee ae
    a[:, [2, 4], t] = a[:, [4, 2], t]
    # 交换ae
    a[:, [3, 4], t] = a[:, [4, 3], t]  # aww aw ap ae aee
    A = np.zeros((count, count))  # 生成10x10矩阵
    A[0:count, 0:a.shape[1]] = a    # 左闭右开
    for i in range(0, count):
        A[i][:] = shift(A[i][:], i-2, cval=0)

    return A, b


def fun():
    # 输入参数
    # 广义对流扩散系数 Gamma_w and Gamma_e
    Gamma = np.array([0.1, 0.1], dtype=np.float64)
    L = 1  # 几何尺寸为1
    crossA = np.array([1, 1], dtype=np.float64)  # 截面积恒定  A_w and A_e

    u = np.array([0.1, 0.1], dtype=np.float64)  # 风速 u_w and u_e
    # 空气密度千克/立方米 rho_w and rho_e
    rho = np.array([1.0, 1.0], dtype=np.float64)
    Ta = 1
    Tb = 0

    dx = np.array([])
    # 一维问题 V=1
    V = np.array(np.ones((count, 1)))

    global numerical_T

    a0 = rho[0]*(L/count)/time_step

    for t in range(1, len(time)):
        t = int(t)
        for i in range(0, count):  # 定时，算区域
            deltaX = distanceOfNodes(L, i+1)
            if i == 0:
                dx = deltaX
            else:
                dx = np.insert(dx, i, values=deltaX, axis=0)
            # V[i] = (dx[i][0]+dx[i][1])/2  # 一维问题V=(dx_W+dx_E)/2
            IMS(Gamma, dx[i], rho, u, crossA, a0, V, L, i, t)
        pass



        A, b = matReshape(Ta, Tb, a[:, :, t], a0, dx, S, V, count, t)
        for i in range(count):
            A[i, abs(A[i, :]) < 1e-10] = 0

        numerical = solve(A, b)  # 数值解
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

    Y = np.row_stack((np.ones((1, len(time)+1)), numerical_T))
    # Y = np.insert(numerical_T, 0, values=np.ones((1,len(time)+1)) ,axis=0)
    # 添加边界温度Tb = 0
    Y = np.row_stack((Y, np.zeros((1, len(time)+1))))

    # 添加边界节点位置Xa = 0
    X = np.insert(sumL, 0, values=0)
    # 添加边界节点位置Xb = L
    X = np.insert(X, len(X), values=L)

    for t in range(1, len(time), 5):
        plt.scatter(X, Y[:, t])
    plt.show()

    return A, b


if __name__ == '__main__':
    A, b = fun()
