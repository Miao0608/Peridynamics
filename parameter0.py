# ！！！写在前面------------------------------------------------------------------------------------------------------------
# 以下代码中可以修改的参数为：
# length_side
# length_x
# length_y
# length_z
# time_total
# number_file_output
# litho #地层划分部分根据实际情况修改函数
# e
# dens
# bulkm
# youngm
# shearm
# cohesion
# angle_in
# waterc
# path
# m1
# m2
# theta
# 其余参数请勿修改！
# PDmodel0.py文件中无需任何修改！
# 将parameter0.py文件中的参数准备好后，直接运行PDmodel0.py文件即可！
# 结果的可视化可以使用paraview软件！

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy import pi
from numpy import sqrt
from time import *
import math

t_start=time()

#-----------------------------------------------------------------------------------------------------------------------

#质点间距
length_side = 10

#地质体尺寸
length_x = 2000 + 4 * length_side
length_y = 7 * length_side
length_z = 500 + length_side + 2 * length_side

particle_x = int(length_x / length_side)
particle_y = int(length_y / length_side)
particle_z = int(length_z / length_side)

particle_total = particle_x * particle_y * particle_z

print('particle_total', particle_total)

volume_particle = length_side ** 3

#-----------------------------------------------------------------------------------------------------------------------

horizon = 3.015 * length_side

#-----------------------------------------------------------------------------------------------------------------------

time_total = 1000 #时间步
time_step = 1

#-----------------------------------------------------------------------------------------------------------------------

number_file_output = 10 #输出文件数量
interval = time_total / number_file_output

#-----------------------------------------------------------------------------------------------------------------------

coord = np.zeros((particle_total, 3))
count1 = 0
for i in range(0, particle_x):
    for i2 in range(0, particle_y):
        for i3 in range(0, particle_z):
            coord[count1, 0] = 0 - 2 * length_side + i * length_side
            coord[count1, 1] = 0 - 2 * length_side + i2 * length_side
            coord[count1, 2] = 0 - 2 * length_side + i3 * length_side
            count1 += 1

particle_total = count1

print('particle_total', particle_total)

keypoint = particle_total-1

#-----------------------------------------------------------------------------------------------------------------------

ID = np.zeros((particle_total, 1))
count1 = 0
for i in range(0, particle_total):
    ID[count1, 0] = i
    count1 += 1

#-----------------------------------------------------------------------------------------------------------------------

#地层分组
#注意：如有基岩，务必设置litho[i, 0] = 10000
litho = np.zeros((particle_total, 1))
for i in range(0, particle_total):
    bb = 25
    if coord[i, 2] <= -2.75 * coord[i, 0] + 2997.48:
        if 450 < coord[i, 2] <= 500:
            litho[i, 0] = 1
        elif 400 < coord[i, 2] <= 450:
            litho[i, 0] = 2
        elif 350 < coord[i, 2] <= 400:
            litho[i, 0] = 1
        elif 300 < coord[i, 2] <= 350:
            litho[i, 0] = 2
        elif 250 < coord[i, 2] <= 300:
            litho[i, 0] = 1
        elif 200 < coord[i, 2] <= 250:
            litho[i, 0] = 2
        elif 150 < coord[i, 2] <= 200:
            litho[i, 0] = 1
        elif 100 < coord[i, 2] <= 150:
            litho[i, 0] = 2
        elif 50 < coord[i, 2] <= 100:
            litho[i, 0] = 1
        else:
            litho[i, 0] = 2
    else:
        if 450 - bb < coord[i, 2] <= 500:
            litho[i, 0] = 1
        elif 400 - bb < coord[i, 2] <= 450 - bb:
            litho[i, 0] = 2
        elif 350 - bb < coord[i, 2] <= 400 - bb:
            litho[i, 0] = 1
        elif 300 - bb < coord[i, 2] <= 350 - bb:
            litho[i, 0] = 2
        elif 250 - bb < coord[i, 2] <= 300 - bb:
            litho[i, 0] = 1
        elif 200 - bb < coord[i, 2] <= 250 - bb:
            litho[i, 0] = 2
        elif 150 - bb < coord[i, 2] <= 200 - bb:
            litho[i, 0] = 1
        elif 100 - bb < coord[i, 2] <= 150 - bb:
            litho[i, 0] = 2
        elif 50 - bb < coord[i, 2] <= 100 - bb:
            litho[i, 0] = 1
        else:
            litho[i, 0] = 2

#-----------------------------------------------------------------------------------------------------------------------

#地层参数设置
#注意：相邻地层的杨氏模量最好不要相差大于10倍，如果必须大于10倍，需设置过渡层，以使杨氏模量平滑过渡

v = np.zeros((particle_total,1))
e = np.zeros((particle_total,1))
dens = np.zeros((particle_total,1))
bulkm = np.zeros((particle_total,1))
youngm = np.zeros((particle_total,1))
shearm = np.zeros((particle_total,1))
cohesion = np.zeros((particle_total,1))
angle_in = np.zeros((particle_total,1))

for i in range(0, particle_total):
    if litho[i, 0] == 1:
        youngm[i, 0] = 8e6 #杨氏模量
        cohesion[i, 0] = 1e4 #内聚力
        angle_in[i, 0] = pi / 6 #内摩擦角
    elif litho[i, 0] == 2:
        youngm[i, 0] = 1e8
        cohesion[i, 0] = 1e2
        angle_in[i, 0] = (40/180) * pi

for i in range(0, particle_total):
    v[i, 0] = 0.25 #泊松比
    e[i, 0] = 0.3 #孔隙比
    dens[i, 0] = 1800 #土体密度
    bulkm[i, 0] = youngm[i, 0] / (3 * (1 - 2 * v[i, 0])) #体积模量
    shearm[i, 0] = youngm[i, 0] / (2 * (1 + v[i, 0])) #剪切模量

# print('bulkm[i, 0],shearm[i, 0]',bulkm[i, 0],shearm[i, 0])

#-----------------------------------------------------------------------------------------------------------------------

waterc  = np.zeros((particle_total,1))
for i in range(0, particle_total):
    if coord[i, 2] <= -2.75 * coord[i, 0] + 2997.48 and 100 < coord[i, 2] <= 450:
        if litho[i, 0] == 2:
            waterc[i, 0] = -75 #地下水位变化
        else:
            waterc[i, 0] = 0
    else:
        waterc[i, 0] = 0

#-----------------------------------------------------------------------------------------------------------------------

output1 = np.zeros((particle_total, 14))
for i in range(0, particle_total):
    output1[i, 0] = ID[i, 0]
    output1[i, 1] = coord[i, 0]
    output1[i, 2] = coord[i, 1]
    output1[i, 3] = coord[i, 2]
    output1[i, 4] = litho[i, 0]
    output1[i, 5] = v[i, 0]
    output1[i, 6] = e[i, 0]
    output1[i, 7] = dens[i, 0]
    output1[i, 8] = bulkm[i, 0]
    output1[i, 9] = youngm[i, 0]
    output1[i, 10] = shearm[i, 0]
    output1[i, 11] = cohesion[i, 0]
    output1[i, 12] = angle_in[i, 0]
    output1[i, 13] = waterc[i, 0]
count1 += 1

path = 'D:/Documents/'
header1 = "ID coord_x coord_y coord_z lithology v e dens bulkm youngm shearm cohesion angle_in groundwater_level_decline"
np.savetxt(path+'parameter0.txt',output1[:,:],header=header1, comments='', fmt='%.6f')

#-----------------------------------------------------------------------------------------------------------------------
# water_pressure = np.zeros((particle_total,1))
# pressc = np.zeros((particle_total,1))
# #
# count1 = 0
# for tt in range(0,time_total):
#     for i in range(0, particle_total):
#         # pressc[i, 0] = 1000 * 9.81 * waterc[i, 0] * tt / time_total
#         if 0 <= coord[i, 2] <= length_z / 2:
#             pressc[i, 0] = 1000 * 9.81 * waterc[i, 0] * tt / time_total
#         else:
#             pressc[i, 0] = 0
#     output2 = np.zeros((particle_total, 2))
#     if tt % 1 == 0:
#         for i in range(0, particle_total):
#             output2[i, 0] = ID[i, 0]
#             output2[i, 1] = pressc[i, 0]
#         count1 += 1
#     header2 = "ID pressc"
#     np.savetxt(path + 'pressc%g.txt' % tt, output2[:, :], header=header2, comments='')

# ----------------------------------------------------------------------------------------------------------------------
sigmaZ = np.zeros((particle_total,1))
sigma1 = np.zeros((particle_total,1))
sigma2 = np.zeros((particle_total,1))
sigmaX = np.zeros((particle_total,1))
sigmaY = np.zeros((particle_total,1))
tauXY = np.zeros((particle_total,1))
tauXZ = np.zeros((particle_total,1))
tauYZ = np.zeros((particle_total,1))
soil_pressure = np.zeros((particle_total,1))

count1 = 0
for i in range(0, particle_total):
    if coord[i, 2] == length_z - 3 * length_side:
        soil_pressure[i, 0] = dens[i, 0] * 9.81 * (1/2 * length_side) * (1 - e[i, 0])
    elif coord[i, 2] < 0:
        soil_pressure[i, 0] = dens[i, 0] * 9.81 * (length_z - 3 * length_side) * (1 - e[i, 0])
    else:
        soil_pressure[i, 0] = dens[i, 0] * 9.81 * (length_z - 3 * length_side - coord[i, 2]) * (1 - e[i, 0])

    m1 = 0.38
    m2 = 0.38
    theta = 0  #math.pi / 6

    # Vertical stress
    sigmaZ[i, 0] = -soil_pressure[i, 0]

    # Principal stress components
    sigma1[i, 0] = sigmaZ[i, 0] * m1
    sigma2[i, 0] = sigmaZ[i, 0] * m2

    # Compute the rotation from principal reference system to x-y reference system
    costh = np.cos(theta)
    sinth = np.sin(theta)
    sigmaX[i, 0] = sigma1[i, 0] * costh ** 2 + sigma2[i, 0] * sinth ** 2
    sigmaY[i, 0] = sigma1[i, 0] * sinth ** 2 + sigma2[i, 0] * costh ** 2
    tauXY[i, 0] = (sigma1[i, 0] - sigma2[i, 0]) * sinth * costh

    # tauXZ and tauYZ are zero by definition (sigma3 and sigmaZ coincide)
    tauXZ[i, 0] = 0
    tauYZ[i, 0] = 0

    # Final stress tensor
    stressTensor = np.array([[sigmaX[i, 0], tauXY[i, 0], tauXZ[i, 0]],
                             [tauXY[i, 0], sigmaY[i, 0], tauYZ[i, 0]],
                             [tauXZ[i, 0], tauYZ[i, 0], sigmaZ[i, 0]]])

print('stressTensor', stressTensor)

output3 = np.zeros((particle_total, 15))
for i in range(0, particle_total):
    output3[i, 0] = ID[i, 0]
    output3[i, 1] = coord[i, 0]
    output3[i, 2] = coord[i, 1]
    output3[i, 3] = coord[i, 2]
    output3[i, 4] = litho[i, 0]
    output3[i, 5] = dens[i, 0]
    output3[i, 6] = sigmaZ[i, 0]
    output3[i, 7] = sigma1[i, 0]
    output3[i, 8] = sigma2[i, 0]
    output3[i, 9] = sigmaX[i, 0]
    output3[i, 10] = sigmaY[i, 0]
    output3[i, 11] = tauXY[i, 0]
    output3[i, 12] = tauXZ[i, 0]
    output3[i, 13] = tauYZ[i, 0]
    output3[i, 14] = soil_pressure[i, 0]
count1 += 1
header3 = "ID coordX coordY coordZ lithology density sigmaZ sigma1 sigma2 sigmaX sigmaY tauXY tauXZ tauYZ soil_pressure"
np.savetxt(path + 'init3DStress.txt', output3[:, :], header=header3, comments='')

t_end=time()
print(t_end-t_start,'Seconds')
print((t_end-t_start)/3600,'Hours')






