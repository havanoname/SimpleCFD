# SIMPLE + 一阶迎风
import numpy as np
# import scipy.linalg

# 几何尺寸与物理参数
L = 2.0  # 长度
A_in = 0.5  # 入口面积
A_out = 0.1  # 出口面积
P_in = 10  # 入口压力
P_out = 0   # 出口压力
rho = 1  # 密度为kg/m3
gamma = 0  # 无摩擦, 黏性扩散项=0
relax_factors = 0.8  # 松弛因子


# 网格信息
nodes_P = 5  # 压力节点数 A B C D E
ConVol_P = nodes_P-1  # 压力控制体
nodes_V = nodes_P - 1  # 速度节点数 1 2 3 4
ConVol_V = nodes_V  # 速度控制体

# 确定压力节点左右距离  A B C D E
# 压力节点到左边节点距离
Distance_PL = np.ones((nodes_P, 1))  # 压力节点左距离
Distance_PL = Distance_PL * (L/ConVol_P)
Distance_PL[0][0] = 0  # 入口节点A的左边距离为0
# Distance_PL[1][0] = Distance_PL[1][0] + (L/ConVol_P)/2  # 注意,由于采用交错网格技术, 压力节点实际采用外节点法, P到W和E节点的距离相同

Distance_PR = np.ones((nodes_P, 1))  # 压力节点右距离
Distance_PR = Distance_PR * (L/ConVol_P)
Distance_PR[nodes_P-1][0] = 0   # 出口节点E的右边距离为0 注意python索引从0开始, 左闭右开
# Distance_PR[nodes_P-2][0] = Distance_PR[nodes_P-2][0] + (L/ConVol_P) / 2  # 同上, 压力节点为外节点法, 左右节点距离相同
# 合并
Distance_P = np.concatenate((Distance_PL, Distance_PR), axis=1)

# 确定速度节点左右距离  1 2 3 4
# 速度节点到左边节点的距离
Distance_VL = np.ones((nodes_V, 1))
Distance_VL = Distance_VL * (L/ConVol_V)  # 注意速度节点都在几何体内部, 边界没有节点!
Distance_VL[0][0] = 0
# 速度节点到右边节点的距离
Distance_VR = np.ones((nodes_V, 1))
Distance_VR = Distance_VR * (L/ConVol_V)
Distance_VR[nodes_V-1][0] = 0
# 合并
Distance_V = np.concatenate((Distance_VL, Distance_VR), axis=1)

# 为场初始化做准备
# 速度节点所在位置的坐标 A为0点, E处为l=2 --> 所以只需用到左边距离distance_VL
l0 = (L / nodes_V) / 2  # 1的坐标为0.25
x_V = np.ones((nodes_V, 1))
x_V = x_V * l0  # 初始化坐标
# 见https://stackoverflow.com/questions/33883758/python-sum-all-previous-values-in-array-at-each-index 注意 matlab也有这种操作, numpy的函数命基本和matlab对应
sumDisVL = np.cumsum(Distance_VL)
for i in range(1, nodes_V):  # 从1开始, 因为0对应1号节点, 在上一步已经初始化为0.25
    x_V[i][0] = x_V[i][0] + sumDisVL[i]  # x_V即为速度节点的坐标

x_P = np.linspace(start=0, stop=L, num=nodes_P)
x_PP = np.expand_dims(x_P, axis=1)  # 升维,保证数组统一

# 计算速度节点所对应的截面积, 用相似三角形
A_xV = A_in + (A_in - A_out) / (0 - L) * x_V
# 计算压力节点所对应的截面积
A_xP = A_in + (A_in - A_out) / (0 - L) * x_P
A_xP = np.expand_dims(A_xP, axis=1)  # 升维,保证数组统一

# 场初始化
# 速度场- 1 2 3 4
m_ = 1  #
u_init = m_ / (rho*A_xV)
# 压力场A B C D E
p_init = np.linspace(start=10, stop=0, num=nodes_P)
p_init = np.expand_dims(p_init, axis=1)  # 升维,保证数组统一

## 正式计算开始 ##
# 速度初始化必要参数--一阶迎风 三点格式
a_v = np.zeros((nodes_V, 3))  # aw ae ap
F_v = np.zeros((nodes_V, 2))  # Fw Fe
D_v = np.zeros((nodes_V, 2))  # Dw De
Su_v = np.zeros((nodes_V, 1))  # 速度网格的Su
d_v = np.zeros((nodes_V, 1))

# 计算速度网格, 这里图方便就不用矩阵运算了, 直接用循环, 虽然计算量可能会偏大
for i in range(0, nodes_V):
    if i != 0 and i != nodes_V-1:  # 计算中间节点 如2 3
        F_v[i][0] = rho * ((u_init[i-1] + u_init[i])/2) * A_xP[i][0]  # Fw
        F_v[i][1] = rho * ((u_init[i] + u_init[i+1])/2) * A_xP[i+1][0]  # Fe
        D_v[i][0] = gamma / (Distance_VL[i][0])  # Dw
        D_v[i][1] = gamma / (Distance_VR[i][0])  # De  gamma=0, 这两个均为0
        # 计算系数a
        a_v[i][0] = D_v[i][0] + max(F_v[i][0], 0)  # aw
        a_v[i][1] = D_v[i][1] + max(0, -F_v[i][1])  # ae
        a_v[i][2] = a_v[i][0] + a_v[i][1] + \
            (F_v[i][1] - F_v[i][0])  # ap = aw + ae + (Fe - Fw)
        Su_v[i] = (p_init[i-1][0] - p_init[i][0]) * \
            A_xV[i]  # 速度网格Su = delta_P * A_xV
        d_v[i] = A_xV[i] / a_v[i][2]  # d = A/ap

    # 处理边界节点 1
    if i == 0:
        u_old = u_init[i][0]
        uA = u_init[i][0] * A_xV[i] / A_xP[i]
        F_v[i][0] = rho * uA * A_xP[i]
        F_v[i][1] = rho * (u_init[i][0] + u_init[i+1][0])/2 * A_xP[i+1][0]
        D_v[i][0] = 0  # Dw
        D_v[i][1] = gamma / (Distance_VR[i][0])
        a_v[i][0] = 0  # 左边已经没有节点
        a_v[i][1] = D_v[i][1] + max(0, -F_v[i][1])
        # 边界的ap为啥这样算我也不知道....注意是 Fe + Fw *1/2(A1/AA)^2
        a_v[i][2] = a_v[i][0] + a_v[i][1] + F_v[i][1] + \
            F_v[i][0] * (A_xV[i]/A_xP[i])**2 / 2
        Su_v[i] = (p_init[i][0] - p_init[i+1][0])*A_xV[i] + \
            F_v[i][0] * (A_xV[i]/A_xP[i])*u_old  # 压力p0-pB
        d_v[i] = A_xV[i] / a_v[i][2]

    # 处理边界节点 4
    if i == nodes_V-1:
        F_v[i][0] = rho * ((u_init[i-1][0] + u_init[i][0])/2) * A_xP[i][0]
        # Fe 直接用速度网格计算..我也不知道为啥
        F_v[i][1] = rho * (u_init[i]) * A_xV[i][0]
        D_v[i][0] = gamma / (Distance_VL[i][0])
        D_v[i][1] = 0
        a_v[i][0] = D_v[i][0] + max(F_v[i][0], 0)
        a_v[i][1] = D_v[i][1] + max(0, -F_v[i][1])
        a_v[i][2] = a_v[i][0] + a_v[i][1] + (F_v[i][1] - F_v[i][0])
        Su_v[i] = (p_init[i-1][0] - p_init[i][0])*A_xV[i]
        d_v[i] = A_xV[i] / a_v[i][2]

# 求解方程
matrix_Av = np.zeros((nodes_V, nodes_V))
# 组装系数矩阵
# python里a_v[:,2]和a_v[:][2]输出不同..我人傻了, a_v[:][2]和a_v[2][:]输出的是第二行
np.fill_diagonal(matrix_Av, a_v[:, 2])  # 将ap填入对角线
rng = np.arange(nodes_V-1)
matrix_Av[rng+1, rng] = -1*a_v[1:nodes_V, 0]  # 将aw填入offest=-1对角线


vector_bv = Su_v
u_ = np.linalg.solve(matrix_Av, vector_bv)  # 记为u*

# 压力修正参数初始化
a_p = np.zeros((nodes_P, 3))  # aW aE aP
F_p = np.zeros((nodes_P, 2))  # FW* FE*
D_p = np.zeros((nodes_P, 2))  # DW DE
Su_p = np.zeros((nodes_P, 1))  # 速度网格的Su
d_p = np.zeros((nodes_P, 1))
b_ = np.zeros((nodes_P, 1))
p_ = np.zeros((nodes_P, 1))  # 记为p'
p_now = np.zeros((nodes_P, 1))  # 记为p*

# 压力修正
for i in range(0, nodes_P):
    if i != 0 and i != nodes_P-1:  # 计算中间节点
        a_p[i, 0] = rho * d_v[i-1, 0] * A_xV[i-1, 0]  # aW = rho * d1 * A1
        # aE = rho * d2 * A2  # 注意压力网格和速度网格差一个标号, 所以这里不用+1
        a_p[i, 1] = rho * d_v[i, 0] * A_xV[i, 0]
        F_p[i, 0] = rho * u_[i-1, 0] * A_xV[i-1, 0]  # FW = rho * u_ * A1
        F_p[i, 1] = rho * u_[i, 0] * A_xV[i, 0]
        D_p[i, 0] = gamma / (Distance_PL[i, 0])  # DW = 0
        D_p[i, 1] = gamma / (Distance_PR[i, 0])  # DE = 0
        a_p[i, 2] = a_p[i, 0] + a_p[i, 1]  # aP = aW + aE
        b_[i, 0] = F_p[i, 0] - F_p[i, 1]

    # 计算边界节点, 书上设p_=0 我也不知道为啥
    if i == 0 or i == nodes_P-1:
        p_[i, 0] = 0

# 求解
matrix_Ap = np.zeros((nodes_P-2, nodes_P-2))  # 减去两个边界节点
np.fill_diagonal(matrix_Ap, a_p[1:nodes_P-1, 2])
rng = np.arange(nodes_P-1-2)  # 减去两个边界点
matrix_Ap[rng+1, rng] = -1*a_p[2:nodes_V, 0]  # 自己看P207页自己悟吧..就是把pB'和pC'移项到offest=-1
matrix_Ap[rng, rng+1] = -1*a_p[1:nodes_V-1, 1]  # 移另一项

vector_bp = b_[1:nodes_P-1,0]
# 针对bp处理边界的A和E
vector_bp[0] = a_p[0,0]*p_[0,0] + vector_bp[0]
vector_bp[len(vector_bp)-1] = a_p[len(a_p)-1,1]*p_[nodes_P-1,0] + vector_bp[len(vector_bp)-1]  # 多维数组len()返回行数 即nodes_P
vector_bp = np.expand_dims(vector_bp, axis=1)  # 升维

p_[1:nodes_P-1] = np.linalg.solve(matrix_Ap, vector_bp)  # 左闭右开

# 计算修正压力
p_now[1:nodes_P-1] = p_init[1:nodes_P-1] + p_[1:nodes_P-1]
# 计算修正速度
delta_p_ = np.zeros((nodes_V,1))
for i in range(0, nodes_V):
    delta_p_[i,0] = p_[i,0]-p_[i+1,0]
u_now = u_ + d_v*delta_p_
# 计算入口A点的修正压力
p_now[0,0] = P_in - 1/2 * rho * u_now[0,0]**2 * (A_xV[0,0]/A_xP[0,0])**2

# 连续性检查
conti_check = rho * u_now * A_xV  # 所有元素一样则是正确的


# 引入松弛因子
u_new = (1-relax_factors) * u_init + relax_factors*u_now
p_new = (1-relax_factors) * p_init + relax_factors*p_now

# 进入下一步迭代