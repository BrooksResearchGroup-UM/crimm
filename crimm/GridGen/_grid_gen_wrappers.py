import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

nd_float_ptr_type = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
# Load the shared library
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'grid_gen.so'))
# Define the argument dtype and return types for the C functions
cdist = lib.calc_pairwise_dist
cdist.restype = None
cdist.argtypes = [
    nd_float_ptr_type, # grid_pos
    nd_float_ptr_type, # coords
    ctypes.c_int, # N_coord
    ctypes.c_int, # N_grid_points
    nd_float_ptr_type # dists (out)
]

gen_elec_grid = lib.gen_elec_grid
gen_elec_grid.restype = None
gen_elec_grid.argtypes = [
    nd_float_ptr_type, # dists
    nd_float_ptr_type, # charges
    ctypes.c_double, # cc_elec
    ctypes.c_double, # rad_dielec_const
    ctypes.c_double, # elec_attr_max
    ctypes.c_double, # elec_attr_max
    ctypes.c_int, # N_coord
    ctypes.c_int, # N_grid_points
    nd_float_ptr_type # elec_grid (out)
]

gen_vdw_grid = lib.gen_vdw_grid
gen_vdw_grid.restype = None
gen_vdw_grid.argtypes = [
    nd_float_ptr_type, # dists
    nd_float_ptr_type, # epsilons
    nd_float_ptr_type, # vdw_rs
    ctypes.c_double, # probe_radius
    ctypes.c_double, # vwd_softcore_max
    ctypes.c_int, # N_coord
    ctypes.c_int, # N_grid_points
    nd_float_ptr_type # vdw_grid (out)
]

def cdist_wrapper(grid_pos, coords):
    N_coord, N_grid_points = coords.shape[0], grid_pos.shape[0]
    dists = np.zeros(N_grid_points*N_coord, dtype=np.double, order='C')
    cdist(
        grid_pos, coords, N_coord, N_grid_points, dists
    )
    return dists.reshape(N_grid_points, N_coord)

def gen_elec_grid_wrapper(
    dists, charges, cc_elec, rad_dielec_const, elec_rep_max, elec_attr_max
):
    N_coord = charges.shape[0]
    assert dists.size%N_coord == 0
    N_grid_points = dists.size//N_coord

    elec_grid = np.zeros(N_grid_points, dtype=np.double, order='C')
    gen_elec_grid(
        dists, charges, cc_elec, rad_dielec_const, elec_rep_max, elec_attr_max,
        N_coord, N_grid_points, elec_grid
    )

    return elec_grid

def gen_vdw_grid_wrapper(
    dists, epsilons, vdw_rs, probe_radius, vwd_softcore_max
):
    N_coord = epsilons.shape[0]
    assert dists.size%N_coord == 0
    N_grid_points = dists.size//N_coord

    vdw_grid = np.zeros(N_grid_points, dtype=np.double, order='C')
    gen_vdw_grid(
        dists, epsilons, vdw_rs, probe_radius, vwd_softcore_max,
        N_coord, N_grid_points, vdw_grid
    )

    return vdw_grid
