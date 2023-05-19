# QUICK格式

# 二阶迎风格式SUD
import numpy as np
from numpy.linalg import *
import math
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift


# count of volume
count = 10

a = np.array(np.zeros((count, 5)))  # aww aw ae aee ap
S = np.array(np.zeros((count, 2)))  # Sc Sp


def QUICK(Gamma, dx, rho, u, A, V, L, i):  # 每一段控制体进行计算
    D = Gamma / dx
    F = rho * u

    alpha = 0
    if u[0] > 0:
        alpha = 1
    if u[1] > 0:
        alpha = 1

    Pe = rho*u/(Gamma*L)

    global a, S
    # a[i, 0] = (D[0] + 1/2 * F[0])*A[0]  # aw
    # a[i, 1] = (D[1] - 1/2 * F[1])*A[1]  # ae
    a[i, 0] = -1/8 * alpha * F[0]  # aww
    a[i, 1] = D[0] + 6/8 * alpha * F[0] + 1/8 * \
        alpha * F[1] + 3/8 * (1-alpha)*F[0]  # aw
    a[i, 2] = D[1] - 3/8 * alpha*F[1] - 6/8 * (1-alpha) * F[1] - 1 / \
        8 * (1-alpha) * F[0]   # ae
    a[i, 3] = 1/8 * (1-alpha) * F[1]  # aee

    # 边界点处理: 边界用三点差分的一阶迎风，发现不好使啊！！
    # if i == 0:  # 第一个节点
    #     a[i, 0] = 0   # aww = 0
    #     a[i, 1] = D[0] + max(F[0], 0)  # aww = Dw + max(Fw , 0)
    #     a[i, 2] = D[1] + max(0, -F[1])  # ae = De + max(0, -Fe)
    #     a[i, 3] = 0  # aee = 0
    # if i == count - 1:  # 最后一个节点
    #     a[i, 0] = 0  # aww
    #     a[i, 1] = D[0] + max(F[0], 0)  # aw
    #     a[i, 2] = D[1] + max(0, -F[1])  # ae = D[e] + max(0, -Fe)
    #     a[i, 3] = 0     # aee = 0

    # 参考https://blog.csdn.net/weixin_42562856/article/details/107323235
    # 参考book: H.Versteeg - An Introduction to Computational Fluid Dynamics  P160-161
    # 使用QUICK格式在边界点的处理
    Da_start = Gamma[0]/dx[1]  # 在第一个节点才会用，所以dx[1] = 0.1
    Db_start = Gamma[1]/dx[0]  # 在最后一个节点才会用，所以dx[0] = 0.1
    if i == 0:  # 第一个节点
        a[i, 0] = 0   # aww = 0
        a[i, 1] = 0  # aw = 0
        # ae = De + 1/3D_{A*} - 3/8Fe.  D_{A*} = gamma/dx[1] PS 是dx[1]不是0
        a[i, 2] = D[1] + 1/3*Da_start - 3/8*F[1]
        a[i, 3] = 0  # aee = 0
        S[i, 0] = 8/3 * Da_start + 2/8*F[1] + F[0]  # Sc
        S[i, 1] = -1 * (8/3 * Da_start + 2/8 * F[1] + F[0])  # Sp
    elif i == 1:
        a[i, 0] = 0
        a[i, 1] = D[0]+7/8*F[0]+1/8*F[1]  # aw
        a[i, 2] = D[1] - 3/8*F[1]  # ae
        a[i, 3] = 0
        S[i, 0] = -1/4 * F[0]
        S[i, 1] = 1/4 * F[0]
    elif i == count - 1:  # 最后一个节点
        a[i, 0] = -1/8*F[0]  # aww
        a[i, 1] = D[0] + 1/3*Db_start + 6/8*F[0]  # aw
        a[i, 2] = 0  # ae
        a[i, 3] = 0     # aee = 0
        S[i, 0] = (8/3 * Db_start - F[1])  # Sc
        S[i, 1] = -(8/3 * Db_start - F[1])  # Sp

    # 前面判断完再计算ap
    a[i, 4] = a[i, 0] + a[i, 1] + a[i, 2] + \
        a[i, 3] + F[0]*A[0] - F[1]*A[1] - S[i, 1] * V[i]  # ap


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


def matReshape(Ta, Tb, a, dx, S, V, count):  # 重组矩阵成为系数矩阵

    # 注意在该函数被调用前，a已经被更改为 aww aw ap ae aee

    b = np.mat(np.zeros((count, 1)))
    b = S[:, 0] * V[0]

    b[0:2] = b[0:2]*Ta  # a_w1*Ta
    b[count-1] = b[count-1]*Tb  # a_en*Tb

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
    A = np.array([1, 1], dtype=np.float64)  # 截面积恒定  A_w and A_e

    u = np.array([2, 2], dtype=np.float64)  # 风速 u_w and u_e
    # 空气密度千克/立方米 rho_w and rho_e
    rho = np.array([1.093, 1.093], dtype=np.float64)
    Ta = 50  # 下端温度
    Tb = 20  # 上端温度

    dx = np.array([])
    # V = np.array(np.zeros((count, 1)))
    # 一维问题 V=1
    V = np.array(np.ones((count, 1)))
    for i in range(0, count):
        deltaX = distanceOfNodes(L, i+1)
        if i == 0:
            dx = deltaX
        else:
            dx = np.insert(dx, i, values=deltaX, axis=0)
        # V[i] = (dx[i][0]+dx[i][1])/2  # 一维问题V=(dx_W+dx_E)/2
        QUICK(Gamma, dx[i], rho, u, A, V, L, i)

    # a的原形式 aww aw ae aee ap
    a[:, 0:3] = -1*a[:, 0:3]  # 对aw和ae取负表示移项
    # 交换ap和ae，使得ap在中间，因为matReshape函数假设ap在中间 aww aw ap aee ae
    a[:, [2, 4]] = a[:, [4, 2]]
    # 交换ae
    a[:, [3, 4]] = a[:, [4, 3]]  # aww aw ap ae aee

    A, b = matReshape(Ta, Tb, a, dx, S, V, count)
    for i in range(count):
        A[i, abs(A[i, :]) < 1e-10] = 0

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
    plt.scatter(sumL, numerical)
    plt.show()

    return A, b


if __name__ == '__main__':
    A, b = fun()
