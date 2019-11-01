from dolfin import *
import numpy as np

def apply_minmod_limiter(u):
    fs = u.function_space()
    mesh = fs.mesh()
    dofmap = fs.dofmap()                                                                         
    for cell in cells(mesh):
        if cell.index() > 0 and cell.index() < mesh.num_cells() - 1:                                                                         
            cell_idx = cell.index()
            dx = 2*cell.circumradius()
            vertex_values = u.vector()[dofmap.cell_dofs(cell_idx)]

            avg_left = np.average(u.vector()[dofmap.cell_dofs(cell_idx-1)])
            avg_this = np.average(vertex_values)
            avg_right = np.average(u.vector()[dofmap.cell_dofs(cell_idx+1)])

            a1 = (vertex_values[1] - vertex_values[0])/dx
            a2 = (avg_this - avg_left)/dx
            a3 = (avg_right - avg_this)/dx

            if np.sign(a1) == np.sign(a2) and np.sign(a2) == np.sign(a3):
                m = np.sign(a1)*min(abs(a1), abs(a2), abs(a3))
            else:
                m = 0.0
            u.vector()[dofmap.cell_dofs(cell_idx)] = np.array([avg_this - m*(0.5*dx), avg_this + m*(0.5*dx)])
