import numpy as np 
import matplotlib.pyplot as plt
import math

#== parameters ===
nx = 5  # 网格单元数
nndoes = nx + 2 # 节点数，含边界

L = 1.0 # 长度，m
gamma = 0.1  #扩散系数 , kg/m.s
phi_a = 1 # 边界A的温度值 
phi_b = 0 # 边界B的温度值 
rho = 1.0 # 密度， kg/m^3
u = 2.5 # 速度，m/s # 0.1 , 2.5
# =========================


#==  x grid ==
dx = L/nx  # 网格间距
print('dx = ',dx)
x = np.zeros(nndoes)  # x网格
x[1:nndoes-1] = np.linspace(dx/2, L-dx/2, nx) # 以边界A为原点创建网格点的坐标值
x[-1] = x[-2] + dx/2 #边界B的坐标值
print('x grid = ', x, '\n')

#==  solution array ==
phi = np.zeros(nndoes)  # 解向量
phi[0] = phi_a # 边界值
phi[-1] = phi_b

DD = gamma / dx  # D
FF = rho * u     # F
Pe = rho * u * dx / gamma      # Peclet number
#== matrix ==
A = np.zeros((nx, nx)) 
b = np.zeros(nx)

#### 内部网格节点  #########
su = 0.0
sp = 0.0
for i in range(1, nx-1):    
    A[i][i-1] = -(DD + FF)
    A[i][i+1] = - DD
    A[i][i] = -(A[i][i-1] + A[i][i+1]) - sp
    b[i] = su

# for boundary A
i = 0  
A[i][i+1] = -DD
su = (2*DD + FF) * phi_a
sp = -(2*DD + FF)
A[i][i] = -A[i][i+1] - sp
b[i] = su

# for boundary B
i = nx-1  
A[i][i-1] = -(DD + FF)
su = 2*DD * phi_b
sp = - 2*DD
A[i][i] = -A[i][i-1] - sp
b[i] = su

print('A = \n', A, '\n')
print('b = \n', np.matrix(b).T ,'\n')

phi_temp = np.linalg.solve(A, b)
print('solution = \n', np.matrix(phi_temp).T, '\n')
phi[1:nndoes-1] = phi_temp

#===== for exact solution ======
xx = np.linspace(0, L, 50, endpoint=True)
exact_solution = np.zeros(50)
for i in range(50):
    exact_solution[i] = (math.exp(rho*u*xx[i] / gamma) -1) / (math.exp(rho*u*L / gamma) -1) * (phi_b - phi_a) + phi_a
    
plt.xlabel('Distance (m)')
plt.ylabel('Phi (C)')
plt.plot(x,phi ,'bs--', label='Numerical (UD)')
plt.plot(xx,exact_solution,'k', label='Exact')
title = 'u= '+str(u)+',  Pe= %.3f'% Pe
plt.title(title.rstrip('0'))
plt.legend()
plt.show()