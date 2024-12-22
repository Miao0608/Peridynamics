import numpy as np
from numpy import sqrt
from time import *
from numba import njit
from numpy import pi
import scipy
import parameter0

#-----------------------------------------------------------------------------------------------------------------------
t_start=time()

# Time
time_total = parameter0.time_total
time_step = parameter0.time_step

number_file_output=parameter0.number_file_output
interval=parameter0.interval

# Geological body
length_side=parameter0.length_side
length_x = parameter0.length_x
length_y = parameter0.length_y
length_z = parameter0.length_z
particle_total=parameter0.particle_total
volume_particle=parameter0.volume_particle

keypoint = parameter0.keypoint

# Horizon
horizon=parameter0.horizon

print('time_step',time_step)

# 读取parameter0.py生成的2个txt文件---------------------------------------------------------------------------------------------------------
num_header_lines = 1
path = parameter0.path
data1 = np.loadtxt(path+'parameter0.txt',skiprows=num_header_lines)

ID = np.zeros((particle_total,1))
coord = np.zeros((particle_total, 3))
litho = np.zeros((particle_total,1))
v = np.zeros((particle_total, 1))
e = np.zeros((particle_total, 1))
dens = np.zeros((particle_total, 1))
bulkm = np.zeros((particle_total, 1))
youngm = np.zeros((particle_total, 1))
shearm = np.zeros((particle_total, 1))
cohesion = np.zeros((particle_total,1))
angle_in = np.zeros((particle_total,1))
waterc = np.zeros((particle_total, 1))
pressc = np.zeros((particle_total,1))
for i in range(0, particle_total):
    ID[i, 0] = data1[i, 0]
    coord[i, 0] = data1[i, 1]
    coord[i, 1] = data1[i, 2]
    coord[i, 2] = data1[i, 3]
    litho[i, 0] = data1[i, 4]
    v[i, 0] = data1[i, 5]
    e[i, 0] = data1[i, 6]
    dens[i, 0] = data1[i, 7]
    bulkm[i, 0] = data1[i, 8]
    youngm[i, 0] = data1[i, 9]
    shearm[i, 0] = data1[i, 10]
    cohesion[i, 0] = data1[i, 11]
    angle_in[i, 0] = data1[i, 12]
    waterc[i, 0] = data1[i, 13]

data3 = np.loadtxt(path+'init3DStress.txt',skiprows=num_header_lines)
sigmaZ_init = np.zeros((particle_total,1))
sigma1_init = np.zeros((particle_total,1))
sigma2_init = np.zeros((particle_total,1))
sigmaX_init = np.zeros((particle_total,1))
sigmaY_init = np.zeros((particle_total,1))
tauXY_init = np.zeros((particle_total,1))
tauXZ_init = np.zeros((particle_total,1))
tauYZ_init = np.zeros((particle_total,1))
for i in range(0, particle_total):
    sigmaX_init[i, 0] = data3[i, 9]
    sigmaY_init[i, 0] = data3[i, 10]
    sigmaZ_init[i, 0] = data3[i, 6]
    tauXY_init[i, 0] = data3[i, 11]
    tauXZ_init[i, 0] = data3[i, 12]
    tauYZ_init[i, 0] = data3[i, 13]

@njit()
def initialize_arrays():
    numx1 = np.zeros((particle_total, 1))
    numx1new = np.zeros((particle_total, 1))
    ncoord = np.zeros((particle_total, 3))
    pforce = np.zeros((particle_total, 4))
    pforceold = np.zeros((particle_total, 3))
    bforce = np.zeros((particle_total, 3))
    acc = np.zeros((particle_total, 3))
    vel = np.zeros((particle_total, 3))
    disp = np.zeros((particle_total, 3))
    dispmath = np.zeros((particle_total, 1))

    pressc = np.zeros((particle_total, 1))
    water_pressure = np.zeros((particle_total, 1))

    wvol = np.zeros((particle_total, 1))
    dilatax1 = np.zeros((particle_total, 1))
    velhalf = np.zeros((particle_total, 3))
    velhalfold = np.zeros((particle_total, 3))
    massvec = np.zeros((particle_total, 3))

    bc = np.zeros((particle_total, 1))

    numx2 = np.zeros((particle_total, 7 * 7 * 5, 1))
    dist = np.zeros((particle_total, 7 * 7 * 5, 1))
    ndist = np.zeros((particle_total, 7 * 7 * 5, 1))
    influc = np.zeros((particle_total, 7 * 7 * 5, 1))
    volcorr = np.zeros((particle_total, 7 * 7 * 5, 1))
    exten = np.zeros((particle_total, 7 * 7 * 5, 1))
    extend = np.zeros((particle_total, 7 * 7 * 5, 2))
    dilatax2 = np.zeros((particle_total, 7 * 7 * 5, 1))

    t_center = np.zeros((particle_total, 3))
    t_neighbor = np.zeros((particle_total, 7 * 7 * 5, 3))
    w_center = np.zeros((particle_total, 3))
    w_neighbor = np.zeros((particle_total, 7 * 7 * 5, 3))
    wt_center = np.zeros((particle_total, 3))
    wt_neighbor = np.zeros((particle_total, 7 * 7 * 5, 3))
    wt = np.zeros((particle_total, 3))

    xishu = np.zeros((particle_total, 7 * 7 * 5, 3))
    xishu2 = np.zeros((particle_total, 3))

    streshe = np.zeros((particle_total, 7 * 7 * 5, 1))
    streten = np.zeros((particle_total, 7 * 7 * 5, 1))
    cristreshe = np.zeros((particle_total, 7 * 7 * 5, 1))
    cristreten = np.zeros((particle_total, 7 * 7 * 5, 1))
    Dongjqml = np.zeros((particle_total, 7 * 7 * 5, 1))

    tau = np.zeros((particle_total, 7 * 7 * 5, 3))
    tau_norm = np.zeros((particle_total, 7 * 7 * 5, 1))

    A0 = np.zeros((particle_total, 7 * 7 * 5, 3))
    A1 = np.zeros((particle_total, 7 * 7 * 5, 3))
    A2 = np.zeros((particle_total, 7 * 7 * 5, 3))
    B0 = np.zeros((particle_total, 7 * 7 * 5, 3))
    B1 = np.zeros((particle_total, 7 * 7 * 5, 3))
    B2 = np.zeros((particle_total, 7 * 7 * 5, 3))

    B0B0 = np.zeros((particle_total, 1))
    B0B1 = np.zeros((particle_total, 1))
    B0B2 = np.zeros((particle_total, 1))
    B1B0 = np.zeros((particle_total, 1))
    B1B1 = np.zeros((particle_total, 1))
    B1B2 = np.zeros((particle_total, 1))
    B2B0 = np.zeros((particle_total, 1))
    B2B1 = np.zeros((particle_total, 1))
    B2B2 = np.zeros((particle_total, 1))
    A0B0 = np.zeros((particle_total, 1))
    A0B1 = np.zeros((particle_total, 1))
    A0B2 = np.zeros((particle_total, 1))
    A1B0 = np.zeros((particle_total, 1))
    A1B1 = np.zeros((particle_total, 1))
    A1B2 = np.zeros((particle_total, 1))
    A2B0 = np.zeros((particle_total, 1))
    A2B1 = np.zeros((particle_total, 1))
    A2B2 = np.zeros((particle_total, 1))

    B0B0old = np.zeros((particle_total, 1))
    B0B1old = np.zeros((particle_total, 1))
    B0B2old = np.zeros((particle_total, 1))
    B1B0old = np.zeros((particle_total, 1))
    B1B1old = np.zeros((particle_total, 1))
    B1B2old = np.zeros((particle_total, 1))
    B2B0old = np.zeros((particle_total, 1))
    B2B1old = np.zeros((particle_total, 1))
    B2B2old = np.zeros((particle_total, 1))
    A0B0old = np.zeros((particle_total, 1))
    A0B1old = np.zeros((particle_total, 1))
    A0B2old = np.zeros((particle_total, 1))
    A1B0old = np.zeros((particle_total, 1))
    A1B1old = np.zeros((particle_total, 1))
    A1B2old = np.zeros((particle_total, 1))
    A2B0old = np.zeros((particle_total, 1))
    A2B1old = np.zeros((particle_total, 1))
    A2B2old = np.zeros((particle_total, 1))

    vector1 = np.zeros((particle_total, 7 * 7 * 5, 3))
    vector2 = np.zeros((particle_total, 7 * 7 * 5, 3))
    cross_product1 = np.zeros((particle_total, 7 * 7 * 5, 3))
    cross_product2 = np.zeros((particle_total, 7 * 7 * 5, 3))
    shape_tensor = np.zeros((particle_total, 1))
    inverse_shape_tensor = np.zeros((particle_total, 1))
    deformation_tensor = np.zeros((particle_total, 1))
    deformation_gradient_tensor = np.zeros((particle_total, 1))

    damage = np.ones((particle_total, 7 * 7 * 5, 1))
    crack = np.zeros((particle_total, 1))

    soil_pressure = np.zeros((particle_total, 1))
    stress_total = np.zeros((particle_total, 1))
    effective_stress = np.zeros((particle_total, 1))
    effective_stress_i = np.zeros((particle_total, 1))
    effective_stress_c = np.zeros((particle_total, 1))

    principal_stress = np.zeros((particle_total, 3))
    shear_stress = np.zeros((particle_total, 3))
    bond_stress = np.zeros((particle_total, 7 * 7 * 5, 1))

    sigmaN = np.zeros((particle_total, 7 * 7 * 5, 1))
    shear_strength = np.zeros((particle_total, 7 * 7 * 5, 1))
    max_tensile_stress = np.zeros((particle_total, 7 * 7 * 5, 1))

    sigmaX_change = np.zeros((particle_total, 1))
    sigmaY_change = np.zeros((particle_total, 1))
    sigmaZ_change = np.zeros((particle_total, 1))
    tauXY_change = np.zeros((particle_total, 1))
    tauYZ_change = np.zeros((particle_total, 1))
    tauXZ_change = np.zeros((particle_total, 1))

    sigmaX_changeold = np.zeros((particle_total, 1))
    sigmaY_changeold = np.zeros((particle_total, 1))
    sigmaZ_changeold = np.zeros((particle_total, 1))
    tauXY_changeold = np.zeros((particle_total, 1))
    tauYZ_changeold = np.zeros((particle_total, 1))
    tauXZ_changeold = np.zeros((particle_total, 1))

    sigmaX_curr = np.zeros((particle_total, 1))
    sigmaY_curr = np.zeros((particle_total, 1))
    sigmaZ_curr = np.zeros((particle_total, 1))
    tauXY_curr = np.zeros((particle_total, 1))
    tauYZ_curr = np.zeros((particle_total, 1))
    tauXZ_curr = np.zeros((particle_total, 1))

    output = np.zeros((particle_total, 21, time_total))

    return numx1, ncoord, pforce, pforceold, bforce, acc, vel, disp, dispmath, pressc, water_pressure, \
        wvol, dilatax1, velhalf, velhalfold, massvec, bc, numx2, dist, ndist, influc, \
        volcorr, exten, extend, dilatax2, streshe, streten, shear_strength, soil_pressure, \
        Dongjqml, cristreshe, cristreten, t_center, t_neighbor, w_center, w_neighbor, \
        wt_center, wt_neighbor, wt, xishu, xishu2, crack, damage, principal_stress, \
        shear_stress, bond_stress, stress_total, effective_stress, effective_stress_i, \
        A0, A1, A2, B0, B1, B2,effective_stress_c, tau_norm,tau, sigmaN, max_tensile_stress, \
        sigmaX_change, sigmaY_change, sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, \
        sigmaX_curr, sigmaY_curr, sigmaZ_curr, tauXY_curr, tauYZ_curr, tauXZ_curr, \
        B0B0,B0B1,B0B2,B1B0,B1B1,B1B2,B2B0,B2B1,B2B2,A0B0,A0B1,A0B2,A1B0,A1B1,A1B2,A2B0,A2B1,A2B2,\
        B0B0old,B0B1old,B0B2old,B1B0old,B1B1old,B1B2old,B2B0old,B2B1old,B2B2old,A0B0old,A0B1old,\
        A0B2old,A1B0old,A1B1old,A1B2old,A2B0old,A2B1old,A2B2old,numx1new,\
        sigmaX_changeold, sigmaY_changeold, sigmaZ_changeold, tauXY_changeold, tauYZ_changeold, tauXZ_changeold,output

@njit()
def neighbors_search(coord, numx2, dist, volcorr, numx1, damage,numx1new):
    for i in range(0, particle_total):
        count1 = 0
        for i2 in range(0, particle_total):
            if i != i2:
                length_bond = sqrt((coord[i, 0] - coord[i2, 0]) ** 2 +
                                   (coord[i, 1] - coord[i2, 1]) ** 2 +
                                   (coord[i, 2] - coord[i2, 2]) ** 2)
                if length_bond <= horizon:
                    numx2[i, count1, 0] = i2
                    dist[i, count1, 0] = length_bond
                    if length_bond <= (horizon - (length_side / 2)):
                        volcorr[i, count1, 0] = 1  # Volume correction factor
                    else:
                        volcorr[i, count1, 0] = (horizon + (
                                    length_side / 2) - length_bond) / length_side  # Volume correction factor
                    count1 += 1
        numx1[i, 0] = count1
        numx1new[i, 0] = numx1[i, 0]
    print('Neighboring particles search completed')

@njit()
def compute_weighted_volume(wvol, influc, dist, volcorr, pforce, numx1):
    for i in range(0, particle_total):
        pforce[i, 0] = 0
        pforce[i, 1] = 0
        pforce[i, 2] = 0

        # Influence factor------------------------------------------------------------------------------------------------------
        for i2 in range(0, int(numx1[i, 0])):
            influc[i, i2, 0] = 1

        # Weighted volume--------------------------------------------------------------------------------------------------
        m = 0
        for i2 in range(0, int(numx1[i, 0])):
            m += (influc[i, i2, 0] * dist[i, i2, 0]) * dist[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
        wvol[i, 0] = m

@njit()
def compute_length_after_deformation(sigmaN, effective_stress, ndist, exten, coord, disp, numx1, numx2, dist, streshe,
                                     streten, shear_strength, soil_pressure, Dongjqml, cristreshe, cristreten, principal_stress):
    for i in range(0, particle_total):
        for i2 in range(0, int(numx1[i, 0])):
            ID_neighbor = int(numx2[i, i2, 0])
            length_bond_new = sqrt((coord[i, 0] + disp[i, 0] - coord[ID_neighbor, 0] - disp[ID_neighbor, 0]) ** 2 +
                                   (coord[i, 1] + disp[i, 1] - coord[ID_neighbor, 1] - disp[ID_neighbor, 1]) ** 2 +
                                   (coord[i, 2] + disp[i, 2] - coord[ID_neighbor, 2] - disp[ID_neighbor, 2]) ** 2)
            ndist[i, i2, 0] = length_bond_new
            # Extension scalar state
            exten[i, i2, 0] = ndist[i, i2, 0] - dist[i, i2, 0]

@njit()
def compute_dilatation(damage,dilatax1, dilatax2, exten, influc, wvol, numx1, dist, volcorr, numx2):
    for i in range(0, particle_total):
        # theta(x)
        for i2 in range(0, int(numx1[i, 0])):
            if damage[i, i2, 0] == 0:
                influc[i, i2, 0] = 0
            dilatax1[i, 0] += ((3 / wvol[i, 0]) * influc[i, i2, 0] * dist[i, i2, 0]) * exten[
                i, i2, 0] * volume_particle * volcorr[i, i2, 0]
        # theta(x')
        for i2 in range(0, int(numx1[i, 0])):  # i的邻域点数量
            if damage[i, i2, 0] == 0:
                influc[i, i2, 0] = 0
            ID_neighbor = int(numx2[i, i2, 0])  # i的领域点编号
            dilatax2[i, i2, 0] += ((3 / wvol[ID_neighbor, 0]) * influc[i, i2, 0] * dist[i, i2, 0]) * exten[
                i, i2, 0] * volume_particle * volcorr[i, i2, 0]

@njit()
def compute_extension_scalar_state(damage, influc, extend, exten, dilatax1, dilatax2, dist, numx1):
    for i in range(0, particle_total):
        # ed(x),ed(x')
        for i2 in range(0, int(numx1[i, 0])):
            extend[i, i2, 0] = exten[i, i2, 0] - ((dilatax1[i, 0] * dist[i, i2, 0]) / 3)  # 中心点i的
            extend[i, i2, 1] = exten[i, i2, 0] - ((dilatax2[i, i2, 0] * dist[i, i2, 0]) / 3)  # 中心点i的邻域点的
@njit()
def compute_scalar_force_state(litho, damage, pforce, t_center, t_neighbor, w_center, w_neighbor, wt_center, wt_neighbor, wt, xishu,
                               coord, disp, ndist, influc, volcorr, pressc, bulkm, shearm, numx1, numx2, dilatax1, dist,
                               wvol, extend, dilatax2):

    for i in range(0, particle_total):
        # t(x),t(x')
        for i2 in range(0, int(numx1[i, 0])):
            ID_neighbor = int(numx2[i, i2, 0])
            if damage[i, i2, 0] == 0:
                influc[i, i2, 0] = 0
            t_center[i, 0] = -3 * (-bulkm[i, 0] * dilatax1[i, 0]) * influc[i, i2, 0] * dist[i, i2, 0] / wvol[i, 0] + \
                             15 * shearm[i, 0] * influc[i, i2, 0] * extend[i, i2, 0] / wvol[i, 0]
            t_neighbor[i, i2, 0] = -3 * (-bulkm[ID_neighbor, 0] * dilatax2[i, i2, 0]) * influc[i, i2, 0] * dist[i, i2, 0] / wvol[ID_neighbor, 0] + \
                                   15 * shearm[ID_neighbor, 0] * influc[i, i2, 0] * extend[i, i2, 1] / wvol[ID_neighbor, 0]

            w_center[i, 0] = -3 * 1 * pressc[i, 0] * influc[i, i2, 0] * dist[i, i2, 0] / wvol[i, 0]
            w_neighbor[i, i2, 0] = -3 * 1 * pressc[ID_neighbor, 0] * influc[i, i2, 0] * dist[i, i2, 0] / wvol[ID_neighbor, 0]

            wt_center[i, 0] = t_center[i, 0] + w_center[i, 0]
            wt_neighbor[i, i2, 0] = t_neighbor[i, i2, 0] + w_neighbor[i, i2, 0]

            wt[i, 0] = (wt_center[i, 0] + wt_neighbor[i, i2, 0]) * volume_particle * volcorr[i, i2, 0]

            # force state T
            xishu[i, i2, 0] = (coord[ID_neighbor, 0] + disp[ID_neighbor, 0] - coord[i, 0] - disp[i, 0]) / ndist[i, i2, 0]
            xishu[i, i2, 1] = (coord[ID_neighbor, 1] + disp[ID_neighbor, 1] - coord[i, 1] - disp[i, 1]) / ndist[i, i2, 0]
            xishu[i, i2, 2] = (coord[ID_neighbor, 2] + disp[ID_neighbor, 2] - coord[i, 2] - disp[i, 2]) / ndist[i, i2, 0]

            pforce[i, 0] += wt[i, 0] * xishu[i, i2, 0]
            pforce[i, 1] += wt[i, 0] * xishu[i, i2, 1]
            pforce[i, 2] += wt[i, 0] * xishu[i, i2, 2]

@njit()
def compute_dynamic_relaxation(velhalf, velhalfold, disp, pforce, pforceold, massvec, tt, time_step, bforce, bc, vel):
    for i in range(0, particle_total):
        bc[i, 0] = 12.0 * youngm[i, 0] / (pi * (horizon ** 4))  #Bond constant
        # 5 is a safety factor
        massvec[i, 0] = 0.25 * time_step ** 2 * (4.0 / 3.0 * pi * horizon ** 3) * bc[i, 0] / length_side #* 5.0
        massvec[i, 1] = 0.25 * time_step ** 2 * (4.0 / 3.0 * pi * horizon ** 3) * bc[i, 0] / length_side #* 5.0
        massvec[i, 2] = 0.25 * time_step ** 2 * (4.0 / 3.0 * pi * horizon ** 3) * bc[i, 0] / length_side #* 5.0

    #Adaptive dynamic relaxation method
    cn = 0.0
    cn1 = 0.0
    cn2 = 0.0

    for i in range(0, particle_total):
        if velhalfold[i, 0] != 0.0:
            cn1 -= disp[i, 0] * disp[i, 0] * (pforce[i, 0] / massvec[i, 0] - pforceold[i, 0] / massvec[i, 0]) / (
                        time_step * velhalfold[i, 0])
        if velhalfold[i, 1] != 0.0:
            cn1 -= disp[i, 1] * disp[i, 1] * (pforce[i, 1] / massvec[i, 1] - pforceold[i, 1] / massvec[i, 1]) / (
                        time_step * velhalfold[i, 1])
        if velhalfold[i, 2] != 0.0:
            cn1 -= disp[i, 2] * disp[i, 2] * (pforce[i, 2] / massvec[i, 2] - pforceold[i, 2] / massvec[i, 2]) / (
                        time_step * velhalfold[i, 2])

        cn2 += disp[i, 0] * disp[i, 0]
        cn2 += disp[i, 1] * disp[i, 1]
        cn2 += disp[i, 2] * disp[i, 2]

    if cn2 != 0.0:
        if cn1 / cn2 > 0.0:
            cn = 2.0 * sqrt(cn1 / cn2)
        else:
            cn = 0.0
    else:
        cn = 0.0

    if cn > 2.0:
        cn = 1.9

    for i in range(0, particle_total):
        if tt == 1:
            velhalf[i, 0] = 1.0 * time_step / massvec[i, 0] * (pforce[i, 0] + bforce[i, 0]) / 2.0
            velhalf[i, 1] = 1.0 * time_step / massvec[i, 1] * (pforce[i, 1] + bforce[i, 1]) / 2.0
            velhalf[i, 2] = 1.0 * time_step / massvec[i, 2] * (pforce[i, 2] + bforce[i, 2]) / 2.0
        else:
            velhalf[i, 0] = ((2.0 - cn * time_step) * velhalfold[i, 0] + 2.0 * time_step / massvec[i, 0] * (
                    pforce[i, 0] + bforce[i, 0])) / (2.0 + cn * time_step)
            velhalf[i, 1] = ((2.0 - cn * time_step) * velhalfold[i, 1] + 2.0 * time_step / massvec[i, 1] * (
                    pforce[i, 1] + bforce[i, 1])) / (2.0 + cn * time_step)
            velhalf[i, 2] = ((2.0 - cn * time_step) * velhalfold[i, 2] + 2.0 * time_step / massvec[i, 2] * (
                    pforce[i, 2] + bforce[i, 2])) / (2.0 + cn * time_step)

        vel[i, 0] = 0.5 * (velhalfold[i, 0] + velhalf[i, 0])
        vel[i, 1] = 0.5 * (velhalfold[i, 1] + velhalf[i, 1])
        vel[i, 2] = 0.5 * (velhalfold[i, 2] + velhalf[i, 2])

        disp[i, 0] += velhalf[i, 0] * time_step
        disp[i, 1] += velhalf[i, 1] * time_step
        disp[i, 2] += velhalf[i, 2] * time_step

        velhalfold[i, 0] = velhalf[i, 0]
        velhalfold[i, 1] = velhalf[i, 1]
        velhalfold[i, 2] = velhalf[i, 2]

        pforceold[i, 0] = pforce[i, 0]
        pforceold[i, 1] = pforce[i, 1]
        pforceold[i, 2] = pforce[i, 2]
        pforce[i, 3] = np.sqrt(pforce[i, 0] ** 2 + pforce[i, 1] ** 2 + pforce[i, 2] ** 2)

    print('disp[i, 2]',disp[keypoint, 2])
    print('minimum disp_z',np.min(disp[:, 2]))

def read_pore_water_pressure_change(water_pressure, tt, particle_total, pressc, dispmath):
    diceng = 150
    for i in range(0, particle_total):
        pressc[i, 0] = 1000 * 9.81 * waterc[i, 0] * tt / time_total
    #     if length_z - diceng <= coord[i, 2] <= length_z:
    #         pressc[i, 0] = 0
    #         # water_pressure[i, 0] = 0
    #     elif length_z - diceng - waterc[i, 0] <= coord[i, 2] < length_z - diceng:
    #         if tt < (length_z - diceng - coord[i, 2]) / (waterc[i, 0] / time_total):
    #             # water_pressure[i, 0] = 1000 * 9.81 * ((length_z - coord[i, 2]) - (waterc[i, 0] / time_total) * tt)
    #             pressc[i, 0] = 1000 * 9.81 * (length_z - diceng - coord[i, 2]) * tt / (((length_z - diceng - coord[i, 2]) / waterc[i, 0]) * time_total)
    #         else:
    #             # water_pressure[i, 0] = 0
    #             pressc[i, 0] = 1000 * 9.81 * (length_z - diceng - coord[i, 2])
    #     else:
    #         # water_pressure[i, 0] = 1000 * 9.81 * ((length_z - coord[i, 2]) - (waterc[i, 0] / time_total) * tt)
    #         pressc[i, 0] = 1000 * 9.81 * waterc[i, 0] * tt / time_total
    #
    #     dispmath[i, 0] = (length_z - length_side) * (-pressc[i, 0]) * (1 / youngm[i, 0]) * (((1 + v[i, 0]) * (1 - 2 * v[i, 0])) / (1 - v[i, 0]))
    # # print('dispmath[i,0]', dispmath[keypoint, 0])

def compute_soil_pressure(water_pressure, stress_total, effective_stress, effective_stress_i, effective_stress_c,
                          particle_total, soil_pressure, dens, length_z, coord, e, principal_stress):
    for i in range(particle_total):
        soil_pressure[i, 0] = dens[i, 0] * 9.81 * (length_z - length_side - coord[i, 2]) * (1 - e[i, 0])
        stress_total[i, 0] = 1000 * 9.81 * (length_z - length_side - coord[i, 2]) * e[i, 0] + soil_pressure[i, 0]
        effective_stress[i, 0] = stress_total[i, 0] - water_pressure[i, 0]
        effective_stress_i[i, 0] = stress_total[i, 0] - 1000 * 9.81 * ((length_z - length_side - coord[i, 2]))
        effective_stress_c[i, 0] = effective_stress[i, 0] - effective_stress_i[i, 0]
    # print('effective_stress_c[i, 0]', effective_stress_c[keypoint, 0])

@njit()
def update_coordinates_after_deformation(ncoord, coord, disp):
    for i in range(0, particle_total):
        ncoord[i, 0] = coord[i, 0] + disp[i, 0]
        ncoord[i, 1] = coord[i, 1] + disp[i, 1]
        ncoord[i, 2] = coord[i, 2] + disp[i, 2]

@njit()
def apply_geological_body_boundary_conditions(litho, disp, coord, length_side, length_x, length_y):
    for i in range(0, particle_total):
        if 0 - 2 * length_side <= coord[i, 0] <= 0 - 2 * length_side + 2 * length_side or \
                length_x - 2 * length_side - 3 * length_side <= coord[i, 0] <= length_x - 2 * length_side:
            disp[i, 0] = 0
        if 0 - 2 * length_side <= coord[i, 1] <= 0 - 2 * length_side + 2 * length_side or \
                length_y - 2 * length_side - 3 * length_side <= coord[i, 1] <= length_y - 2 * length_side:
            disp[i, 1] = 0
        if 0 - 2 * length_side <= coord[i, 2] <= 0 - 2 * length_side + 2 * length_side:
            disp[i, 0] = 0
            disp[i, 1] = 0
            disp[i, 2] = 0
        if litho[i, 0] == 10000: #专为基岩设置
            disp[i, 0] = 0
            disp[i, 1] = 0
            disp[i, 2] = 0

@njit()
def compute_stress(numx1new,crack,damage, sigmaX_change, sigmaY_change, sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, sigmaX_curr,
                   sigmaY_curr, sigmaZ_curr, tauXY_curr, tauYZ_curr, tauXZ_curr, streshe, ndist, dist, max_tensile_stress,
                   sigmaN, tau_norm, tau,shear_strength, effective_stress,numx1,numx2,coord,ncoord,A0,A1,A2,B0,B1,B2,influc,
                   volcorr,principal_stress, shear_stress, xishu, bond_stress,sigmaX_init, sigmaY_init, sigmaZ_init,
                   tauXY_init, tauYZ_init, tauXZ_init,B0B0,B0B1,B0B2,B1B0,B1B1,B1B2,B2B0,B2B1,B2B2,A0B0,A0B1,A0B2,A1B0,
                   A1B1,A1B2,A2B0,A2B1,A2B2,B0B0old,B0B1old,B0B2old,B1B0old,B1B1old,B1B2old,B2B0old,B2B1old,B2B2old,A0B0old,
                   A0B1old,A0B2old,A1B0old,A1B1old,A1B2old,A2B0old,A2B1old,A2B2old,sigmaX_changeold, sigmaY_changeold, sigmaZ_changeold, tauXY_changeold, tauYZ_changeold, tauXZ_changeold):

    for i in range(0, particle_total):
        B0B0[i,0] = 0
        B0B1[i,0] = 0
        B0B2[i,0] = 0
        B1B0[i,0] = 0
        B1B1[i,0] = 0
        B1B2[i,0] = 0
        B2B0[i,0] = 0
        B2B1[i,0] = 0
        B2B2[i,0] = 0
        A0B0[i,0] = 0
        A0B1[i,0] = 0
        A0B2[i,0] = 0
        A1B0[i,0] = 0
        A1B1[i,0] = 0
        A1B2[i,0] = 0
        A2B0[i,0] = 0
        A2B1[i,0] = 0
        A2B2[i,0] = 0

        for i2 in range(0, int(numx1[i, 0])):
            ID_neighbor = int(numx2[i, i2, 0])

            A0[i, i2, 0] = ncoord[ID_neighbor, 0] - ncoord[i, 0]
            A1[i, i2, 0] = ncoord[ID_neighbor, 1] - ncoord[i, 1]
            A2[i, i2, 0] = ncoord[ID_neighbor, 2] - ncoord[i, 2]
            B0[i, i2, 0] = coord[ID_neighbor, 0] - coord[i, 0]
            B1[i, i2, 0] = coord[ID_neighbor, 1] - coord[i, 1]
            B2[i, i2, 0] = coord[ID_neighbor, 2] - coord[i, 2]

            if damage[i, i2, 0] == 0:
                influc[i, i2, 0] = 1e-10

            #张量积Tensor product，在前为竖，在后为横  (以下B0B0等的值均为中心点的，此处考虑了近场域内所有点对中心点的影响)
            # (p-x)圈乘(p-x)
            B0B0[i,0] += B0[i, i2, 0] * B0[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B0B1[i,0] += B0[i, i2, 0] * B1[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B0B2[i,0] += B0[i, i2, 0] * B2[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B1B0[i,0] += B1[i, i2, 0] * B0[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B1B1[i,0] += B1[i, i2, 0] * B1[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B1B2[i,0] += B1[i, i2, 0] * B2[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B2B0[i,0] += B2[i, i2, 0] * B0[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B2B1[i,0] += B2[i, i2, 0] * B1[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            B2B2[i,0] += B2[i, i2, 0] * B2[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            # Y<p-x>圈乘(p-x)
            A0B0[i,0] += A0[i, i2, 0] * B0[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A0B1[i,0] += A0[i, i2, 0] * B1[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A0B2[i,0] += A0[i, i2, 0] * B2[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A1B0[i,0] += A1[i, i2, 0] * B0[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A1B1[i,0] += A1[i, i2, 0] * B1[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A1B2[i,0] += A1[i, i2, 0] * B2[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A2B0[i,0] += A2[i, i2, 0] * B0[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A2B1[i,0] += A2[i, i2, 0] * B1[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            A2B2[i,0] += A2[i, i2, 0] * B2[i, i2, 0] * influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]

            B0B0old[i, 0] = B0B0[i, 0]
            B0B1old[i, 0] = B0B1[i, 0]
            B0B2old[i, 0] = B0B2[i, 0]
            B1B0old[i, 0] = B1B0[i, 0]
            B1B1old[i, 0] = B1B1[i, 0]
            B1B2old[i, 0] = B1B2[i, 0]
            B2B0old[i, 0] = B2B0[i, 0]
            B2B1old[i, 0] = B2B1[i, 0]
            B2B2old[i, 0] = B2B2[i, 0]
            # Y<p-x>圈乘(p-x)
            A0B0old[i, 0] = A0B0[i, 0]
            A0B1old[i, 0] = A0B1[i, 0]
            A0B2old[i, 0] = A0B2[i, 0]
            A1B0old[i, 0] = A1B0[i, 0]
            A1B1old[i, 0] = A1B1[i, 0]
            A1B2old[i, 0] = A1B2[i, 0]
            A2B0old[i, 0] = A2B0[i, 0]
            A2B1old[i, 0] = A2B1[i, 0]
            A2B2old[i, 0] = A2B2[i, 0]

        #nonlocal shape tensor
        matrix1 = np.array([[B0B0[i,0], B0B1[i,0], B0B2[i,0]],
                           [B1B0[i,0], B1B1[i,0], B1B2[i,0]],
                           [B2B0[i,0], B2B1[i,0], B2B2[i,0]]], dtype=np.float64)
        matrix2 = np.array([[A0B0[i,0], A0B1[i,0], A0B2[i,0]],
                           [A1B0[i,0], A1B1[i,0], A1B2[i,0]],
                           [A2B0[i,0], A2B1[i,0], A2B2[i,0]]], dtype=np.float64)

        inverse_matrix1 = np.linalg.inv(matrix1)
        F_tensor = np.dot(matrix2, inverse_matrix1)

        # 计算应变张量
        epsilon = 0.5 * (np.dot(F_tensor.T, F_tensor) - np.eye(3))

        strain_xx = epsilon[0][0]
        strain_yy = epsilon[1][1]
        strain_zz = epsilon[2][2]
        strain_xy = epsilon[0][1]
        strain_yz = epsilon[1][2]
        strain_xz = epsilon[0][2]

        fuction_strainstress1 = ((1 - v[i, 0]) * youngm[i, 0]) / ((1 - 2 * v[i, 0]) * (1 + v[i, 0]))
        fuction_strainstress2 = (v[i, 0] * youngm[i, 0]) / ((1 - 2 * v[i, 0]) * (1 + v[i, 0]))
        matrix_strainstress = np.array([[fuction_strainstress1, fuction_strainstress2, fuction_strainstress2, 0, 0, 0],
                            [fuction_strainstress2, fuction_strainstress1, fuction_strainstress2, 0, 0, 0],
                            [fuction_strainstress2, fuction_strainstress2, fuction_strainstress1, 0, 0, 0],
                            [0, 0, 0, 2 * shearm[i, 0], 0, 0],
                            [0, 0, 0, 0, 2 * shearm[i, 0], 0],
                            [0, 0, 0, 0, 0, 2 * shearm[i, 0]]], dtype=np.float64)
        matrix_strain = np.array([[strain_xx],
                            [strain_yy],
                            [strain_zz],
                            [strain_xy],
                            [strain_yz],
                            [strain_xz]], dtype=np.float64)
        matrix_stresschange = np.dot(matrix_strainstress, matrix_strain) #6*1的矩阵
        if 0 - 2 * length_side <= coord[i, 2] <= 0 - 2 * length_side + 2 * length_side:
            sigmaX_change[i, 0] = 0
            sigmaY_change[i, 0] = 0
            sigmaZ_change[i, 0] = 0
            tauXY_change[i, 0] = 0  # xy
            tauYZ_change[i, 0] = 0  # yz
            tauXZ_change[i, 0] = 0  # xz
        else:
            sigmaX_change[i, 0] = matrix_stresschange[0][0]
            sigmaY_change[i, 0] = matrix_stresschange[1][0]
            sigmaZ_change[i, 0] = matrix_stresschange[2][0]
            tauXY_change[i, 0] = matrix_stresschange[3][0]  # xy
            tauYZ_change[i, 0] = matrix_stresschange[4][0]  # yz
            tauXZ_change[i, 0] = matrix_stresschange[5][0]  # xz

        sigmaX_changeold[i, 0] = sigmaX_change[i, 0]
        sigmaY_changeold[i, 0] = sigmaY_change[i, 0]
        sigmaZ_changeold[i, 0] = sigmaZ_change[i, 0]
        tauXY_changeold[i, 0] = tauXY_change[i, 0]
        tauYZ_changeold[i, 0] = tauYZ_change[i, 0]
        tauXZ_changeold[i, 0] = tauXZ_change[i, 0]

        matrix_stresschange_update = np.array([[sigmaX_change[i, 0]],
                                            [sigmaY_change[i, 0]],
                                            [sigmaZ_change[i, 0]],
                                            [tauXY_change[i, 0]],
                                            [tauYZ_change[i, 0]],
                                            [tauXZ_change[i, 0]]], dtype=np.float64)

        #sigmaX, sigmaY, sigmaZ, tauXY, tauYZ, tauXZ
        initstress_xx = sigmaX_init[i, 0]
        initstress_yy = sigmaY_init[i, 0]
        initstress_zz = sigmaZ_init[i, 0]
        initstress_xy = tauXY_init[i, 0]
        initstress_yz = tauYZ_init[i, 0]
        initstress_xz = tauXZ_init[i, 0]

        matrix_init3Dstress = np.array([[initstress_xx],
                            [initstress_yy],
                            [initstress_zz],
                            [initstress_xy],
                            [initstress_yz],
                            [initstress_xz]])
        matrix_currentstress = matrix_stresschange_update + matrix_init3Dstress
        #应力
        if crack[i, 0] == 1:
            sigmaX_curr[i, 0] = 0
            sigmaY_curr[i, 0] = 0
            sigmaZ_curr[i, 0] = 0
            tauXY_curr[i, 0] = 0
            tauYZ_curr[i, 0] = 0
            tauXZ_curr[i, 0] = 0
        else:
            sigmaX_curr[i, 0] = matrix_currentstress[0][0]
            sigmaY_curr[i, 0] = matrix_currentstress[1][0]
            sigmaZ_curr[i, 0] = matrix_currentstress[2][0]
            tauXY_curr[i, 0] = matrix_currentstress[3][0]  # xy
            tauYZ_curr[i, 0] = matrix_currentstress[4][0]  # yz
            tauXZ_curr[i, 0] = matrix_currentstress[5][0]  # xz

        #虚拟键上的应力
        for i2 in range(0, int(numx1[i, 0])):
            ID_neighbor = int(numx2[i, i2, 0])
            if damage[i, i2, 0] == 1:
                bond_stress[i, i2, 0] = ((sigmaX_curr[i, 0] * xishu[i, i2, 0] + tauXY_curr[i, 0] * xishu[i, i2, 1] + \
                                          tauXZ_curr[i, 0] * xishu[i, i2, 2]) + \
                                         (sigmaX_curr[ID_neighbor, 0] * xishu[i, i2, 0] + tauXY_curr[ID_neighbor, 0] *
                                          xishu[i, i2, 1] + tauXZ_curr[ID_neighbor, 0] * xishu[i, i2, 2])) / 2
                bond_stress[i, i2, 1] = ((tauXY_curr[i, 0] * xishu[i, i2, 0] + sigmaY_curr[i, 0] * xishu[i, i2, 1] + \
                                          tauYZ_curr[i, 0] * xishu[i, i2, 2]) + \
                                         (tauXY_curr[ID_neighbor, 0] * xishu[i, i2, 0] + sigmaY_curr[ID_neighbor, 0] *
                                          xishu[i, i2, 1] + tauYZ_curr[ID_neighbor, 0] * xishu[i, i2, 2])) / 2
                bond_stress[i, i2, 2] = ((tauXZ_curr[i, 0] * xishu[i, i2, 0] + tauYZ_curr[i, 0] * xishu[i, i2, 1] + \
                                          sigmaZ_curr[i, 0] * xishu[i, i2, 2]) + \
                                         (tauXZ_curr[ID_neighbor, 0] * xishu[i, i2, 0] + tauYZ_curr[ID_neighbor, 0] * xishu[
                                             i, i2, 1] + sigmaZ_curr[ID_neighbor, 0] * xishu[i, i2, 2])) / 2
            #法向应力
            sigmaN[i, i2, 0] = bond_stress[i, i2, 0] * xishu[i, i2, 0] + bond_stress[i, i2, 1] * xishu[i, i2, 1] +bond_stress[i, i2, 2] * xishu[i, i2, 2]
            #剪切应力
            tau[i, i2, 0] = bond_stress[i, i2, 0] - sigmaN[i, i2, 0] * xishu[i, i2, 0]
            tau[i, i2, 1] = bond_stress[i, i2, 1] - sigmaN[i, i2, 0] * xishu[i, i2, 1]
            tau[i, i2, 2] = bond_stress[i, i2, 2] - sigmaN[i, i2, 0] * xishu[i, i2, 2]
            tau_norm[i, i2, 0] = sqrt(tau[i, i2, 0]**2 + tau[i, i2, 1]**2 + tau[i, i2, 2]**2)
            #抗剪强度
            shear_strength[i, i2, 0] = -sigmaN[i, i2, 0] * np.tan((angle_in[i, 0] + angle_in[ID_neighbor, 0]) / 2) + (
                    cohesion[i, 0] + cohesion[ID_neighbor, 0]) / 2  # soil_pressure[i, 0]

        #计算局部损伤值
        count2 = 0
        fenzi = 0
        fenmu = 0
        if numx1[i, 0] != 0:
            for i2 in range(0, int(numx1[i, 0])):
                ID_neighbor = int(numx2[i, i2, 0])  # (cohesion[i, 0] + cohesion[ID_neighbor, 0]) / 2
                if (tau_norm[i, i2, 0] > shear_strength[i, i2, 0] or sigmaN[i, i2, 0] > 0 or damage[i, i2, 0] == 0):

                    damage[i, i2, 0] = 0
                    influc[i, i2, 0] = 0
                    count2 += 1
            #     fenzi += influc[i, i2, 0] * damage[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            #     fenmu += influc[i, i2, 0] * volume_particle * volcorr[i, i2, 0]
            # crack[i, 0] = 1 - fenzi / fenmu
            crack[i, 0] = count2 / numx1[i, 0]
            numx1new[i, 0] = numx1[i, 0] - count2
    print('maximum crack', np.max(crack[:, 0]))
    print('minimum numx1new',np.min(numx1new[:, 0]))

@njit()
def output_results(ID, sigmaX_init, sigmaY_init, sigmaZ_init, tauXY_init, tauYZ_init, tauXZ_init,sigmaX_change, sigmaY_change,
                   sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, sigmaX_curr, sigmaY_curr, sigmaZ_curr, tauXY_curr,
                   tauYZ_curr, tauXZ_curr,streshe,pressc, output, ncoord, pforce, acc, vel, disp, dispmath, t_center, w_center,
                   wt, count1, interval, tt, time_total, number_file_output, principal_stress,bond_stress,crack):
    if tt % 1 == 0:
        for i in range(0, particle_total):
            output[i, 0, count1] = ID[i, 0]
            output[i, 1, count1] = ncoord[i, 0]
            output[i, 2, count1] = ncoord[i, 1]
            output[i, 3, count1] = ncoord[i, 2]
            output[i, 4, count1] = litho[i, 0]
            output[i, 5, count1] = pressc[i, 0]
            output[i, 6, count1] = disp[i, 0]
            output[i, 7, count1] = disp[i, 2]
            output[i, 8, count1] = crack[i, 0]
            output[i, 9, count1] = sigmaX_init[i, 0]
            output[i, 10, count1] = sigmaZ_init[i, 0]
            output[i, 11, count1] = sigmaX_change[i, 0]
            output[i, 12, count1] = sigmaZ_change[i, 0]
            output[i, 13, count1] = sigmaX_curr[i, 0]
            output[i, 14, count1] = sigmaZ_curr[i, 0]
            output[i, 15, count1] = dispmath[i, 0]
            output[i, 16, count1] = disp[i, 1]
            output[i, 17, count1] = pforce[i, 0]
            output[i, 18, count1] = pforce[i, 1]
            output[i, 19, count1] = pforce[i, 2]
            output[i, 20, count1] = tauXZ_curr[i, 0]
        count1 += 1
    # print(tt)

# @njit()
def time_iteration_loop(ID, coord, numx2, dist, volcorr, numx1, ncoord, pforce, pforceold, bforce, acc, vel, disp, dispmath,
                        pressc, water_pressure, wvol, dilatax1, velhalf, velhalfold, massvec, bc, t_center, t_neighbor, w_center,
                        w_neighbor, wt_center, wt_neighbor, wt, xishu, xishu2, output, influc, ndist, exten, dilatax2, extend,
                        bulkm, shearm, youngm, waterc, v,A0, A1, A2, B0, B1, B2, streshe, streten, shear_strength, soil_pressure,
                        Dongjqml,cristreshe, cristreten, damage, principal_stress, shear_stress, bond_stress, crack, stress_total,
                        effective_stress, effective_stress_i, effective_stress_c,sigmaX_init, sigmaY_init, sigmaZ_init,
                        tauXY_init, tauYZ_init, tauXZ_init,tau_norm,tau,sigmaN, max_tensile_stress,sigmaX_change, sigmaY_change,
                        sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, sigmaX_curr, sigmaY_curr, sigmaZ_curr,
                        tauXY_curr, tauYZ_curr, tauXZ_curr,B0B0,B0B1,B0B2,B1B0,B1B1,B1B2,B2B0,B2B1,B2B2,A0B0,A0B1,A0B2,
                        A1B0,A1B1,A1B2,A2B0,A2B1,A2B2,B0B0old,B0B1old,B0B2old,B1B0old,B1B1old,B1B2old,B2B0old,B2B1old,
                        B2B2old,A0B0old,A0B1old,A0B2old,A1B0old,A1B1old,A1B2old,A2B0old,A2B1old,A2B2old,numx1new,
                        sigmaX_changeold, sigmaY_changeold, sigmaZ_changeold, tauXY_changeold, tauYZ_changeold, tauXZ_changeold):
    count1 = 0
    for tt in range(0, time_total):
        compute_weighted_volume(wvol, influc, dist, volcorr, pforce, numx1)
        compute_length_after_deformation(sigmaN, effective_stress, ndist, exten, coord, disp, numx1, numx2, dist, streshe,
                                         streten, shear_strength, soil_pressure, Dongjqml, cristreshe, cristreten,
                                         principal_stress)
        compute_dilatation(damage, dilatax1, dilatax2, exten, influc, wvol, numx1, dist, volcorr, numx2)
        compute_extension_scalar_state(damage, influc,extend, exten, dilatax1, dilatax2, dist, numx1)
        compute_scalar_force_state(litho, damage, pforce, t_center, t_neighbor, w_center, w_neighbor, wt_center, wt_neighbor,
                                   wt, xishu, coord, disp, ndist, influc, volcorr, pressc, bulkm, shearm, numx1, numx2,
                                   dilatax1, dist, wvol, extend, dilatax2)
        compute_dynamic_relaxation(velhalf, velhalfold, disp, pforce, pforceold, massvec, tt, time_step, bforce, bc,
                                   vel)

        apply_geological_body_boundary_conditions(litho, disp, coord, length_side, length_x, length_y)
        read_pore_water_pressure_change(water_pressure, tt, particle_total, pressc, dispmath)
        compute_soil_pressure(water_pressure, stress_total, effective_stress, effective_stress_i, effective_stress_c,
                              particle_total, soil_pressure, dens, length_z, coord, e, principal_stress)
        update_coordinates_after_deformation(ncoord, coord, disp)

        compute_stress(numx1new,crack,damage, sigmaX_change, sigmaY_change, sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, sigmaX_curr,
                       sigmaY_curr, sigmaZ_curr, tauXY_curr, tauYZ_curr, tauXZ_curr, streshe, ndist, dist, max_tensile_stress,
                       sigmaN, tau_norm, tau,shear_strength, effective_stress,numx1,numx2,coord,ncoord,A0,A1,A2,B0,B1,B2,influc,
                       volcorr,principal_stress, shear_stress, xishu, bond_stress,sigmaX_init, sigmaY_init, sigmaZ_init,
                       tauXY_init, tauYZ_init, tauXZ_init,B0B0,B0B1,B0B2,B1B0,B1B1,B1B2,B2B0,B2B1,B2B2,A0B0,A0B1,A0B2,A1B0,
                       A1B1,A1B2,A2B0,A2B1,A2B2,B0B0old,B0B1old,B0B2old,B1B0old,B1B1old,B1B2old,B2B0old,B2B1old,B2B2old,A0B0old,
                   A0B1old,A0B2old,A1B0old,A1B1old,A1B2old,A2B0old,A2B1old,A2B2old,sigmaX_changeold, sigmaY_changeold,
                       sigmaZ_changeold, tauXY_changeold, tauYZ_changeold, tauXZ_changeold)

        output_results(ID, sigmaX_init, sigmaY_init, sigmaZ_init, tauXY_init, tauYZ_init, tauXZ_init,sigmaX_change, sigmaY_change,
                       sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, sigmaX_curr, sigmaY_curr, sigmaZ_curr, tauXY_curr,
                       tauYZ_curr, tauXZ_curr,streshe,pressc, output, ncoord, pforce, acc, vel, disp, dispmath, t_center, w_center,
                       wt, count1, interval, tt, time_total, number_file_output, principal_stress,bond_stress,crack)
        count1 += 1
        print(tt)
    return output

# @njit()
def main():
    numx1, ncoord, pforce, pforceold, bforce, acc, vel, disp, dispmath, pressc, water_pressure, \
        wvol, dilatax1, velhalf, velhalfold, massvec, bc, numx2, dist, ndist, influc, \
        volcorr, exten, extend, dilatax2, streshe, streten, shear_strength, soil_pressure, \
        Dongjqml, cristreshe, cristreten, t_center, t_neighbor, w_center, w_neighbor, \
        wt_center, wt_neighbor, wt, xishu, xishu2, crack, damage, principal_stress, \
        shear_stress, bond_stress, stress_total, effective_stress, effective_stress_i, \
        A0, A1, A2, B0, B1, B2,effective_stress_c, tau_norm, tau, sigmaN, max_tensile_stress, \
        sigmaX_change, sigmaY_change, sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, \
        sigmaX_curr, sigmaY_curr, sigmaZ_curr, tauXY_curr, tauYZ_curr, tauXZ_curr,B0B0,B0B1,B0B2,\
        B1B0,B1B1,B1B2,B2B0,B2B1,B2B2,A0B0,A0B1,A0B2,A1B0,A1B1,A1B2,A2B0,A2B1,A2B2,B0B0old,B0B1old,\
        B0B2old,B1B0old,B1B1old,B1B2old,B2B0old,B2B1old,B2B2old,A0B0old,A0B1old,A0B2old,A1B0old,\
        A1B1old,A1B2old,A2B0old,A2B1old,A2B2old,numx1new,sigmaX_changeold, sigmaY_changeold, sigmaZ_changeold, tauXY_changeold, tauYZ_changeold, tauXZ_changeold,output = initialize_arrays()
    neighbors_search(coord, numx2, dist, volcorr, numx1,damage,numx1new)
    output = time_iteration_loop(ID, coord, numx2, dist, volcorr, numx1, ncoord, pforce, pforceold, bforce, acc, vel, disp, dispmath,
                                 pressc, water_pressure, wvol, dilatax1, velhalf, velhalfold, massvec, bc, t_center, t_neighbor, w_center,
                                 w_neighbor, wt_center, wt_neighbor, wt, xishu, xishu2, output, influc, ndist, exten, dilatax2, extend,
                                 bulkm, shearm, youngm, waterc, v,A0, A1, A2, B0, B1, B2, streshe, streten, shear_strength, soil_pressure,
                                 Dongjqml,cristreshe, cristreten, damage, principal_stress, shear_stress, bond_stress, crack, stress_total,
                                 effective_stress, effective_stress_i, effective_stress_c,sigmaX_init, sigmaY_init, sigmaZ_init, tauXY_init,
                                 tauYZ_init, tauXZ_init,tau_norm,tau,sigmaN,max_tensile_stress,sigmaX_change, sigmaY_change,
                                 sigmaZ_change, tauXY_change, tauYZ_change, tauXZ_change, sigmaX_curr, sigmaY_curr, sigmaZ_curr,
                                 tauXY_curr, tauYZ_curr, tauXZ_curr,B0B0,B0B1,B0B2,B1B0,B1B1,B1B2,B2B0,B2B1,B2B2,A0B0,A0B1,A0B2,
                        A1B0,A1B1,A1B2,A2B0,A2B1,A2B2,B0B0old,B0B1old,B0B2old,B1B0old,B1B1old,B1B2old,B2B0old,B2B1old,
                        B2B2old,A0B0old,A0B1old,A0B2old,A1B0old,A1B1old,A1B2old,A2B0old,A2B1old,A2B2old,numx1new,sigmaX_changeold, sigmaY_changeold, sigmaZ_changeold, tauXY_changeold, tauYZ_changeold, tauXZ_changeold)
    print('Time stepping loop completed')
    return output

output=main()

#-----------------------------------------------------------------------------------------------------------------------
header = 'ID coord_X coord_Y coord_Z lithology pressc displacement_X displacement_Z crack ' \
         'sigmaX_initial sigmaZ_initial sigmaX_change sigmaZ_change sigmaX_current sigmaZ_current dispmath ' \
         'displacement_Y pforceX pforceY pforceZ tauXZ_current'  #displacement_X
for i in range(0,time_total):
    if (i+1) % (time_total/number_file_output) == 0:
        np.savetxt(path+'file%g.txt'%i, output[:,:,i], header=header, comments='')

#-----------------------------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# x1 = []
# x2 = []
# y1 = []
# y2 = []
# y3 = []
# for tt in range(0, time_total):
#     x1_0 = tt
#     x1.append(x1_0)
#     y1_0 = output[keypoint, 8, tt]
#     y1.append(y1_0)
#     y2_0 = output[keypoint, 7, tt]
#     y2.append(y2_0)
#     y3_0 = output[keypoint, 15, tt]
#     y3.append(y3_0)
#
# plt.plot(x1, y1, label='crack', marker='o', linestyle='-')
# plt.title('crack')
# plt.xlabel('Time steps')
# plt.ylabel('crack')
# plt.legend()
# plt.savefig(path+'crack.png')
# plt.show()
#
# plt.plot(x1, y2, label='Displacement_Zaxis_model', marker='o', linestyle='-')
# plt.plot(x1, y3, label='Displacement_Zaxis_math', marker='x', linestyle='-')
# plt.title('Displacement_Zaxis')
# plt.xlabel('Time steps')
# plt.ylabel('Displacement')
# plt.legend()
# plt.savefig(path+'Disp.png')
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------
t_end=time()
print(t_end-t_start,'Seconds')
print((t_end-t_start)/60,'Minutes')
print((t_end-t_start)/3600,'Hours')
