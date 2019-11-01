from dolfin import *
import numpy as np

def set_bdry_value(u_list, lbdry_list, rbdry_list):
    V = u_list[0].function_space()
    dof_coordinates = V.tabulate_dof_coordinates()
    lbdry_idx = np.where(dof_coordinates.min() == dof_coordinates)[0]
    rbdry_idx = np.where(dof_coordinates.max() == dof_coordinates)[0]

    for idx in range(len(u_list)):
        #u_list[idx].vector()[lbdry_idx] = lbdry_list[idx]
        u_list[idx].vector()[rbdry_idx] = rbdry_list[idx]
