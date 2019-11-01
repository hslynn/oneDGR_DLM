from dolfin import *
from global_def import *

import scipy.integrate

def path_point(u_L, u_R, s):
    return (1-s)*u_L + s*u_R

def get_S(var_list, mark = None, bdry = None):
    g00_L, g01_L, g11_L = [var('+') for var in var_list[:3]]
    g00_R, g01_R, g11_R = [var('-') for var in var_list[:3]]
    if mark == 'r':
        g00_L, g01_L, g11_L = [var for var in var_list[:3]]
        g00_R, g01_R, g11_R = [var for var in bdry[:3]]
    elif mark == 'l':
        g00_L, g01_L, g11_L = [var for var in bdry[:3]]
        g00_R, g01_R, g11_R = [var for var in var_list[:3]]


    #left side
    invg00_L = g11_L/(g00_L*g11_L-g01_L*g01_L)
    invg01_L = -g01_L/(g00_L*g11_L-g01_L*g01_L)
    invg11_L = g00_L/(g00_L*g11_L-g01_L*g01_L)

    lapse_L = pow(-invg00_L, -0.5)
    shift_L = -invg01_L/invg00_L
    gamma11_L = 1/g11_L

    #right side
    invg00_R = g11_R/(g00_R*g11_R-g01_R*g01_R)
    invg01_R = -g01_R/(g00_R*g11_R-g01_R*g01_R)
    invg11_R = g00_R/(g00_R*g11_R-g01_R*g01_R)

    lapse_R = pow(-invg00_R, -0.5)
    shift_R = -invg01_R/invg00_R
    gamma11_R = 1/g11_R

    speeds = [-(1+paragamma1)*shift_L, -(1+paragamma1)*shift_R, 
            -shift_L-lapse_L*pow(gamma11_L, 0.5), -shift_L+lapse_L*pow(gamma11_L, 0.5),
            -shift_R-lapse_R*pow(gamma11_R, 0.5), -shift_R+lapse_R*pow(gamma11_R, 0.5)]
    return [min_UFL(speeds), max_UFL(speeds)]

def min_UFL(ufl_list):
    if len(ufl_list) == 1:
        return ufl_list[0]
    else: 
        min_temp = min_UFL(ufl_list[:-1])
        return conditional(lt(ufl_list[-1], min_temp), ufl_list[-1], min_temp)

def max_UFL(ufl_list):
    if len(ufl_list) == 1:
        return ufl_list[0]
    else: 
        max_temp = max_UFL(ufl_list[:-1])
        return conditional(gt(ufl_list[-1], max_temp), ufl_list[-1], max_temp)

def get_path_integration_list(var_list, bdry_mark = None, bdry_list = None):
    var_L_list = [var('+') for var in var_list]
    var_R_list = [var('-') for var in var_list]
    if bdry_mark == 'r':
        var_L_list = var_list
        var_R_list = bdry_list
    elif bdry_mark == 'l':
        var_L_list = bdry_list
        var_R_list = var_list

    g00_L, g01_L, g11_L = var_L_list[:3]
    Pi00_L, Pi01_L, Pi11_L = var_L_list[3:6]
    Phi00_L, Phi01_L, Phi11_L = var_L_list[6:9]

    g00_R, g01_R, g11_R = var_R_list[:3]
    Pi00_R, Pi01_R, Pi11_R = var_R_list[3:6]
    Phi00_R, Phi01_R, Phi11_R = var_R_list[6:9]

    def get_integration_func(idx):
        def func(s):
            g00_s, g01_s, g11_s = [path_point(var_L_list[idx], var_R_list[idx], s) for idx in range(3)] 

            invg00_s = g11_s/(g00_s*g11_s-g01_s*g01_s)
            invg01_s = -g01_s/(g00_s*g11_s-g01_s*g01_s)
            invg11_s = g00_s/(g00_s*g11_s-g01_s*g01_s)

            lapse_s = pow(-invg00_s, -0.5)
            shift_s = -invg01_s/invg00_s
            gamma11_s = 1/g11_s

            if idx == 0:
                value = -Constant(1+paragamma1)*shift_s*(g00_R-g00_L)
            elif idx == 1:
                value = -Constant(1+paragamma1)*shift_s*(g01_R-g01_L)
            elif idx == 2:
                value = -Constant(1+paragamma1)*shift_s*(g11_R-g11_L)
            elif idx == 3:
                value = (-paragamma1*paragamma2*shift_s)*(g00_R-g00_L) + (-shift_s)*(Pi00_R-Pi00_L) \
                    + lapse_s*gamma11_s*(Phi00_R - Phi00_L) 
            elif idx == 4:
                value = (-paragamma1*paragamma2*shift_s)*(g01_R-g01_L) + (-shift_s)*(Pi01_R-Pi01_L) \
                    + lapse_s*gamma11_s*(Phi01_R - Phi01_L) 
            elif idx == 5:
                value = (-paragamma1*paragamma2*shift_s)*(g11_R-g11_L) + (-shift_s)*(Pi11_R-Pi11_L) \
                    + lapse_s*gamma11_s*(Phi11_R - Phi11_L) 
            elif idx == 6:
                value = (-paragamma2*lapse_s)*(g00_R-g00_L) + lapse_s*(Pi00_R-Pi00_L) + (-shift_s)*(Phi00_R-Phi00_L)
            elif idx == 7:
                value = (-paragamma2*lapse_s)*(g01_R-g01_L) + lapse_s*(Pi01_R-Pi01_L) + (-shift_s)*(Phi01_R-Phi01_L)
            elif idx == 8:
                value = (-paragamma2*lapse_s)*(g11_R-g11_L) + lapse_s*(Pi11_R-Pi11_L) + (-shift_s)*(Phi11_R-Phi11_L)
            return value
        return func
    func_list = [get_integration_func(idx) for idx in range(len(var_list))]
    return [scipy.integrate.fixed_quad(func, 0, 1)[0] for func in func_list]

def get_NCP_flux_list(var_list, path_integration_list, bdry_mark = None, bdry_list = None):
    S_L, S_R = get_S(var_list, mark = bdry_mark, bdry = bdry_list)
    
    flux_list = []
    for idx in range(len(var_list)):
        U_L = var_list[idx]('+')
        U_R = var_list[idx]('-')
        if bdry_mark == 'r':
            U_L = var_list[idx]
            U_R = bdry_list[idx]
        elif bdry_mark == 'l':
            U_L = bdry_list[idx]
            U_R = var_list[idx]


        U_star = (S_R*U_R - S_L*U_L- path_integration_list[idx])/(S_R - S_L)

        flux = conditional(gt(S_R, 0), 0.5*(S_R*U_star + S_L*U_star - S_L*U_L - S_R*U_R), \
                    0.5*path_integration_list[idx])
        flux_list.append(flux)
    return flux_list

def get_variational_forms(u1, v, var_list, auxi_list, src_list, exact_var_list):
    mesh = var_list[0].function_space().mesh()
    n = FacetNormal(mesh)

    path_integration_list = get_path_integration_list(var_list)
    NCP_flux_list = get_NCP_flux_list(var_list, path_integration_list)

    path_integration_list_right_bdry = get_path_integration_list(var_list, bdry_mark = 'r', bdry_list = exact_var_list)
    NCP_flux_list_right_bdry = get_NCP_flux_list(var_list, path_integration_list_right_bdry, 
                bdry_mark = 'r', bdry_list = exact_var_list)

    path_integration_list_left_bdry = get_path_integration_list(var_list, bdry_mark = 'l', bdry_list = var_list)
    NCP_flux_list_left_bdry = get_NCP_flux_list(var_list, path_integration_list_left_bdry, 
                bdry_mark = 'l', bdry_list = var_list)

    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]

    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]

    F_g00 = (u1 + (-(1+paragamma1)*shift*g00.dx(0)))*v*dx \
            - src_list[0]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[0]*dS \
            + avg(v)*path_integration_list[0]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[0], -NCP_flux_list_left_bdry[0])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[0], path_integration_list_left_bdry[0])*ds

    F_g01 = (u1 + (-(1+paragamma1)*shift*g01.dx(0)))*v*dx \
            - src_list[1]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[1]*dS \
            + avg(v)*path_integration_list[1]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[1], -NCP_flux_list_left_bdry[1])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[1], path_integration_list_left_bdry[1])*ds

    F_g11 = (u1 + (-(1+paragamma1)*shift*g11.dx(0)))*v*dx \
            - src_list[2]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[2]*dS \
            + avg(v)*path_integration_list[2]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[2], -NCP_flux_list_left_bdry[2])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[2], path_integration_list_left_bdry[2])*ds


    F_Pi00 = (u1 + (-paragamma1*paragamma2*shift*g00.dx(0)) + (-shift)*Pi00.dx(0) + lapse*gamma11*Phi00.dx(0))*v*dx\
            - src_list[3]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[3]*dS \
            + avg(v)*path_integration_list[3]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[3], -NCP_flux_list_left_bdry[3])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[3], path_integration_list_left_bdry[3])*ds

    F_Pi01 = (u1 + (-paragamma1*paragamma2*shift*g01.dx(0)) + (-shift)*Pi01.dx(0) + lapse*gamma11*Phi01.dx(0))*v*dx\
            - src_list[4]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[4]*dS \
            + avg(v)*path_integration_list[4]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[4], -NCP_flux_list_left_bdry[4])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[4], path_integration_list_left_bdry[4])*ds

    F_Pi11 = (u1 + (-paragamma1*paragamma2*shift*g11.dx(0)) + (-shift)*Pi11.dx(0) + lapse*gamma11*Phi11.dx(0))*v*dx\
            - src_list[5]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[5]*dS \
            + avg(v)*path_integration_list[5]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[5], -NCP_flux_list_left_bdry[5])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[5], path_integration_list_left_bdry[5])*ds


    F_Phi00 = (u1 + (-paragamma2*lapse*g00.dx(0)) + lapse*Pi00.dx(0) + (-shift)*Phi00.dx(0))*v*dx\
            - src_list[6]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[6]*dS \
            + avg(v)*path_integration_list[6]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[6], -NCP_flux_list_left_bdry[6])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[6], path_integration_list_left_bdry[6])*ds

    F_Phi01 = (u1 + (-paragamma2*lapse*g01.dx(0)) + lapse*Pi01.dx(0) + (-shift)*Phi01.dx(0))*v*dx\
            - src_list[7]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[7]*dS \
            + avg(v)*path_integration_list[7]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[7], -NCP_flux_list_left_bdry[7])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[7], path_integration_list_left_bdry[7])*ds

    F_Phi11 = (u1 + (-paragamma2*lapse*g11.dx(0)) + lapse*Pi11.dx(0) + (-shift)*Phi11.dx(0))*v*dx\
            - src_list[8]*v*dx \
            + (v('+')-v('-'))*NCP_flux_list[8]*dS \
            + avg(v)*path_integration_list[8]*dS \
            + v*conditional(gt(n[0], 0), NCP_flux_list_right_bdry[8], -NCP_flux_list_left_bdry[8])*ds \
            + 0.5*v*conditional(gt(n[0], 0), path_integration_list_right_bdry[8], path_integration_list_left_bdry[8])*ds

    F_list = [F_g00, F_g01,F_g11, F_Pi00,F_Pi01,F_Pi11,F_Phi00,F_Phi01,F_Phi11]
    return F_list
