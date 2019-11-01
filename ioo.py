from dolfin import *
import numpy as np

def read_var_from_files(var_list, folder):
    for idx in range(len(var_list)):
        var = var_list[idx]
        ufile = HDF5File(MPI.comm_world, folder+var.name()+'.hdf5', 'r')
        ufile.read(var, var.name()) 
        ufile.close()

def write_seqs_to_file(fn, seqs):
    with open(fn, 'w') as f:
        for idx_time in range(len(seqs[0])):
            li = []
            for idx_obj in range(len(seqs)):
                li.append(str(seqs[idx_obj][idx_time])) 
            f.write(' '.join(li)+'\n')

def read_seqs_from_file(fn, seqs):
    with open(fn, 'r') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split(" ")
        for idx in range(len(seqs)):
            seqs[idx].append(float(values[idx]))


def plot_function(func, plt_obj, if_abs=False, yscale=None):
    mesh = func.function_space().mesh()
    num_cells = mesh.num_cells()
    for idx in range(num_cells):
        cell = Cell(mesh, idx)
        verts =  cell.get_vertex_coordinates()
        value_left = np.zeros(1, dtype=np.float64)
        value_right = np.zeros(1, dtype=np.float64)
        func.eval_cell(value_left, np.array([verts[0]]), cell)
        func.eval_cell(value_right, np.array([verts[1]]), cell)

        x_array = np.linspace(verts[0], verts[1], 2)
        y_array = np.array([0.0 for x in x_array])
        y_array[0] = value_left
        y_array[-1] = value_right
        if if_abs:
            plt_obj.plot(x_array, abs(y_array), 'b')
        else:
            plt_obj.plot(x_array, y_array, 'b')
    if yscale:
        plt_obj.set_yscale(yscale)
       
def plot_function_dif(f1, f2, min_r, max_r, ax):    
    mesh = f1.function_space().mesh()
    num_cells = mesh.num_cells()
    for idx in range(num_cells):
        cell = Cell(mesh, idx)
        verts =  cell.get_vertex_coordinates()
        if verts[0] >= max_r:
            break
        elif verts[1] <= min_r:
            continue
        value1_left = np.zeros(1, dtype=np.float64)
        value1_right = np.zeros(1, dtype=np.float64)
        f1.eval_cell(value1_left, np.array([verts[0]]), cell)
        f1.eval_cell(value1_right, np.array([verts[1]]), cell)

        value2_left = np.zeros(1, dtype=np.float64)
        value2_right = np.zeros(1, dtype=np.float64)
        f2.eval_cell(value2_left, np.array([verts[0]]), cell)
        f2.eval_cell(value2_right, np.array([verts[1]]), cell)

        x_array = np.linspace(verts[0], verts[1], 2)
        y_array = np.array([0.0 for x in x_array])
        y_array[0] = value1_left - value2_left
        y_array[-1] = value1_right - value2_right
        ax.plot(x_array, y_array, 'b')

