"""
handle boundary conditons
"""

import numpy as np
from dolfin import *
from global_def import *
import function

def apply_bdry_conditions(var_list, rhs_list, dt):
    mesh = var_list[0].function_space().mesh()
    rbdry_coord = mesh.coordinates().max()
  
    rbdry_value_list = [var(rbdry_coord) for var in var_list]
    g00_rbdry, g01_rbdry, g11_rbdry = rbdry_value_list[:3]
    Pi00_rbdry, Pi01_rbdry, Pi11_rbdry = rbdry_value_list[3:6]
    Phi00_rbdry, Phi01_rbdry, Phi11_rbdry = rbdry_value_list[6:9]

    rhs_value_list = [rhs(rbdry_coord) for rhs in rhs_list]
    rhs_g00, rhs_g01, rhs_g11 = rhs_value_list[:3]
    rhs_Pi00, rhs_Pi01, rhs_Pi11 = rhs_value_list[3:6]
    rhs_Phi00, rhs_Phi01, rhs_Phi11 = rhs_value_list[6:9]

    g11_t0_rbdry = rbdry_value_list[2] - rhs_value_list[2]*dt
    b = pow(1/g11_t0_rbdry, 0.5)

    Phi00_t0_rbdry, Phi01_t0_rbdry, Phi11_t0_rbdry = [rbdry_value_list[idx] - rhs_value_list[idx]*dt for idx in [6, 7, 8]]

    g00_rbdry = g00_rbdry - 0*dt
    g01_rbdry = g01_rbdry - 0*dt
    g11_rbdry = g11_rbdry - 0*dt

    Pi00_rbdry = Pi00_rbdry - (-paragamma2/2 * rhs_g00 + 1/2*rhs_Pi00 - b/2 * rhs_Phi00 + 0.25*b**3*Phi00_t0_rbdry*rhs_g11)*dt
    Pi01_rbdry = Pi01_rbdry - (-paragamma2/2 * rhs_g01 + 1/2*rhs_Pi01 - b/2 * rhs_Phi01 + 0.25*b**3*Phi01_t0_rbdry*rhs_g11)*dt
    Pi11_rbdry = Pi11_rbdry - (-paragamma2/2 * rhs_g11 + 1/2*rhs_Pi11 - b/2 * rhs_Phi11 + 0.25*b**3*Phi11_t0_rbdry*rhs_g11)*dt

    Phi00_rbdry = Phi00_rbdry - (paragamma2/(2*b) * rhs_g00 - 1/(2*b)*rhs_Pi00 + 1/2 * rhs_Phi00 - 0.25*b**2*Phi00_t0_rbdry*rhs_g11)*dt
    Phi01_rbdry = Phi01_rbdry - (paragamma2/(2*b) * rhs_g01 - 1/(2*b)*rhs_Pi01 + 1/2 * rhs_Phi01 - 0.25*b**2*Phi01_t0_rbdry*rhs_g11)*dt
    Phi11_rbdry = Phi11_rbdry - (paragamma2/(2*b) * rhs_g11 - 1/(2*b)*rhs_Pi11 + 1/2 * rhs_Phi11 - 0.25*b**2*Phi11_t0_rbdry*rhs_g11)*dt

    rbdry_value_list = [g00_rbdry, g01_rbdry, g11_rbdry, 
            Pi00_rbdry, Pi01_rbdry, Pi11_rbdry,
            Phi00_rbdry, Phi01_rbdry, Phi11_rbdry]

    #left bdry
    lbdry_coord = mesh.coordinates().min()
    lbdry_value_list = [var(lbdry_coord) for var in var_list]
    g00_lbdry, g01_lbdry, g11_lbdry = lbdry_value_list[:3]
    Pi00_lbdry, Pi01_lbdry, Pi11_lbdry = lbdry_value_list[3:6]
    Phi00_lbdry, Phi01_lbdry, Phi11_lbdry = lbdry_value_list[6:9]

    rhs_value_list = [rhs(lbdry_coord) for rhs in rhs_list]
    rhs_g00, rhs_g01, rhs_g11 = rhs_value_list[:3]
    rhs_Pi00, rhs_Pi01, rhs_Pi11 = rhs_value_list[3:6]
    rhs_Phi00, rhs_Phi01, rhs_Phi11 = rhs_value_list[6:9]

    g11_t0_lbdry = lbdry_value_list[2] - rhs_value_list[2]*dt
    b = pow(1/g11_t0_lbdry, 0.5)

    Phi00_t0_lbdry, Phi01_t0_lbdry, Phi11_t0_lbdry = [lbdry_value_list[idx] - rhs_value_list[idx]*dt for idx in [6, 7, 8]]

    g00_lbdry = g00_lbdry - 0*dt
    g01_lbdry = g01_lbdry - 0*dt
    g11_lbdry = g11_lbdry - 0*dt

    Pi00_lbdry = Pi00_lbdry - (-0.5*paragamma2 * rhs_g00 - 0.25*b**3*Phi00_t0_lbdry*rhs_g11 + 0.5*rhs_Pi00 + b/2 * rhs_Phi00)*dt
    Pi01_lbdry = Pi01_lbdry - (-0.5*paragamma2 * rhs_g01 - 0.25*b**3*Phi01_t0_lbdry*rhs_g11 + 0.5*rhs_Pi01 + b/2 * rhs_Phi01)*dt
    Pi11_lbdry = Pi11_lbdry - (-0.5*paragamma2 * rhs_g11 - 0.25*b**3*Phi11_t0_lbdry*rhs_g11 + 0.5*rhs_Pi11 + b/2 * rhs_Phi11)*dt

    Phi00_lbdry = Phi00_lbdry - (-paragamma2/(2*b) * rhs_g00 - 0.25*b**2*Phi00_t0_lbdry*rhs_g11 + 1/(2*b)*rhs_Pi00 + 0.5*rhs_Phi00)*dt
    Phi01_lbdry = Phi01_lbdry - (-paragamma2/(2*b) * rhs_g01 - 0.25*b**2*Phi01_t0_lbdry*rhs_g11 + 1/(2*b)*rhs_Pi01 + 0.5*rhs_Phi01)*dt
    Phi11_lbdry = Phi11_lbdry - (-paragamma2/(2*b) * rhs_g11 - 0.25*b**2*Phi11_t0_lbdry*rhs_g11 + 1/(2*b)*rhs_Pi11 + 0.5*rhs_Phi11)*dt

    lbdry_value_list = [g00_lbdry, g01_lbdry, g11_lbdry, 
            Pi00_lbdry, Pi01_lbdry, Pi11_lbdry,
            Phi00_lbdry, Phi01_lbdry, Phi11_lbdry]

    function.set_bdry_value(var_list, lbdry_value_list, rbdry_value_list)

