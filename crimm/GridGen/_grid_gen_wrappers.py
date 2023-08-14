import os
import warnings
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from crimm.Data.constants import CC_ELEC_CHARMM as CC_ELEC
from crimm.Utils.cuda_info import is_cuda_available

nd_float_ptr_type = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")

class GridCompEngine:
    """Wrapper for the C++ grid generation library."""
    def __init__(self, backend='cpu'):
        self.lib = None
        self._allowed_backends = ['cpu', 'cuda']
        self._backend = None
        self._cdist = None
        self._gen_elec_grid = None
        self._gen_vdw_grid = None
        self.set_backend(backend)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend not in self._allowed_backends:
            raise ValueError(
                f'Backend must be one of {self._allowed_backends}'
            )
        self.set_backend(backend)

    def set_backend(self, backend):
        """Set the backend for the grid generation routines."""
        if backend == 'cuda':
            if is_cuda_available():
                self.lib = ctypes.CDLL(
                    os.path.join(os.path.dirname(__file__), 'grid_gen.cuda.so')
                )
                self._backend = 'cuda'
            else:
                warnings.warn(
                    'CUDA backend requested, but CUDA not available. '
                    'Use CPU backend instead.'
                )
                self.lib = ctypes.CDLL(
                    os.path.join(os.path.dirname(__file__), 'grid_gen.so')
                )
                self._backend = 'cpu'
        else:
            self.lib = ctypes.CDLL(
                os.path.join(os.path.dirname(__file__), 'grid_gen.so')
            )
            self._backend = 'cpu'

        # Define the argument dtype and return types for the C functions
        self._cdist = self.lib.calc_pairwise_dist
        self._cdist.restype = None
        self._cdist.argtypes = [
            nd_float_ptr_type, # grid_pos
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_coord
            ctypes.c_int, # N_grid_points
            nd_float_ptr_type # dists (out)
        ]

        self._gen_elec_grid = self.lib.gen_elec_grid
        self._gen_elec_grid.restype = None
        self._gen_elec_grid.argtypes = [
            nd_float_ptr_type, # dists
            nd_float_ptr_type, # charges
            ctypes.c_double, # CC_ELEC
            ctypes.c_double, # rad_dielec_const
            ctypes.c_double, # elec_attr_max
            ctypes.c_double, # elec_attr_max
            ctypes.c_int, # N_coord
            ctypes.c_int, # N_grid_points
            nd_float_ptr_type # elec_grid (out)
        ]

        self._gen_vdw_grid = self.lib.gen_vdw_grid
        self._gen_vdw_grid.restype = None
        self._gen_vdw_grid.argtypes = [
            nd_float_ptr_type, # dists
            nd_float_ptr_type, # epsilons
            nd_float_ptr_type, # vdw_rs
            ctypes.c_double, # probe_radius
            ctypes.c_double, # vwd_softcore_max
            ctypes.c_int, # N_coord
            ctypes.c_int, # N_grid_points
            nd_float_ptr_type # vdw_grid (out)
        ]

        self._gen_all_grids = self.lib.gen_all_grids
        self._gen_all_grids.restype = None
        self._gen_all_grids.argtypes = [
            nd_float_ptr_type, # grid_pos
            nd_float_ptr_type, # coords
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # epsilons
            nd_float_ptr_type, # vdw_rs
            ctypes.c_double, # CC_ELEC
            ctypes.c_double, # rad_dielec_const
            ctypes.c_double, # elec_rep_max
            ctypes.c_double, # elec_attr_max
            ctypes.c_double, # probe_radius
            ctypes.c_double, # vwd_softcore_max
            ctypes.c_int, # N_coord
            ctypes.c_int, # N_grid_points
            nd_float_ptr_type, # dists (out)
            nd_float_ptr_type, # elec_grid (out)
            nd_float_ptr_type # vdw_grid (out)
        ]

    def cdist(self, grid_pos, coords):
        """Compute the pairwise distances between grid points and coordinates."""
        N_coord, N_grid_points = coords.shape[0], grid_pos.shape[0]
        dists = np.zeros(N_grid_points*N_coord, dtype=np.double, order='C')
        self._cdist(
            # Ensure contiguity of the arrays to pass to the C function
            np.ascontiguousarray(grid_pos), np.ascontiguousarray(coords),
            N_coord, N_grid_points, dists
        )
        return dists.reshape(N_grid_points, N_coord)

    def gen_elec_grid(
        self, dists, charges, rad_dielec_const, elec_rep_max, elec_attr_max
    ):
        """Generate the electrostatic grid."""
        N_coord = charges.shape[0]
        assert dists.size%N_coord == 0
        N_grid_points = dists.size//N_coord

        elec_grid = np.zeros(N_grid_points, dtype=np.double, order='C')
        self._gen_elec_grid(
            dists, charges, CC_ELEC, rad_dielec_const, elec_rep_max, elec_attr_max,
            N_coord, N_grid_points, elec_grid
        )

        return elec_grid

    def gen_vdw_grid(
        self, dists, epsilons, vdw_rs, probe_radius, vwd_softcore_max
    ):
        """Generate the van der Waals grid."""
        N_coord = epsilons.shape[0]
        assert dists.size%N_coord == 0
        N_grid_points = dists.size//N_coord

        vdw_grid = np.zeros(N_grid_points, dtype=np.double, order='C')
        self._gen_vdw_grid(
            dists, epsilons, vdw_rs, probe_radius, vwd_softcore_max,
            N_coord, N_grid_points, vdw_grid
        )

        return vdw_grid

    def gen_all_grids(
            self, grid_pos, coords, charges, epsilons, vdw_rs, 
            rad_dielec_const, elec_rep_max, elec_attr_max, probe_radius,
            vwd_softcore_max
        ):
        """Generate the electrostatic and van der Waals grids."""
        N_coord, N_grid_points = coords.shape[0], grid_pos.shape[0]
        dists = np.zeros(N_grid_points*N_coord, dtype=np.double, order='C')
        elec_grid = np.zeros(N_grid_points, dtype=np.double, order='C')
        vdw_grid = np.zeros(N_grid_points, dtype=np.double, order='C')
        self._gen_all_grids(
            # Ensure contiguity of the arrays to pass to the C function
            np.ascontiguousarray(grid_pos), np.ascontiguousarray(coords),
            charges, epsilons, vdw_rs, CC_ELEC, rad_dielec_const,
            elec_rep_max, elec_attr_max, probe_radius, vwd_softcore_max,
            N_coord, N_grid_points, dists, elec_grid, vdw_grid
        )
        return dists.reshape(N_grid_points, N_coord), elec_grid, vdw_grid