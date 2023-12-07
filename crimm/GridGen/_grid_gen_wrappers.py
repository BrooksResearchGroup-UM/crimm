import os
import warnings
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from crimm.Data.constants import CC_ELEC_CHARMM as CC_ELEC
from crimm.Utils.cuda_info import is_cuda_available

nd_float_ptr_type = ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")

class ReceptorGridCompEngine:
    """Wrapper for the C++ receptor grid generation library."""
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
                    os.path.join(os.path.dirname(__file__), 'grid_gen_simple.cuda.so')
                )
                self._backend = 'cuda'
            else:
                warnings.warn(
                    'CUDA backend requested, but CUDA not available. '
                    'Use CPU backend instead.'
                )
                self.lib = ctypes.CDLL(
                    os.path.join(os.path.dirname(__file__), 'grid_gen_simple.so')
                )
                self._backend = 'cpu'
        else:
            self.lib = ctypes.CDLL(
                os.path.join(os.path.dirname(__file__), 'grid_gen_simple.so')
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

        self._gen_all_grids = self.lib.gen_all_grids
        self._gen_all_grids.restype = None
        self._gen_all_grids.argtypes = [
            nd_float_ptr_type, # grid_pos
            nd_float_ptr_type, # coords
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # epsilons
            nd_float_ptr_type, # vdw_rs
            ctypes.c_float, # CC_ELEC
            ctypes.c_float, # rad_dielec_const
            ctypes.c_float, # elec_rep_max
            ctypes.c_float, # elec_attr_max
            ctypes.c_float, # vwd_rep_max
            ctypes.c_float, # vwd_attr_max
            ctypes.c_int, # N_coord
            ctypes.c_int, # N_grid_points
            nd_float_ptr_type, # elec_grid (out)
            nd_float_ptr_type, # vdw_grid_attr (out)
            nd_float_ptr_type # vdw_grid_rep (out)
        ]

    def cdist(self, grid_pos, coords):
        """Compute the pairwise distances between grid points and coordinates."""
        N_coord, N_grid_points = coords.shape[0], grid_pos.shape[0]
        dists = np.zeros(N_grid_points*N_coord, dtype=np.float32, order='C')
        self._cdist(
            # Ensure contiguity of the arrays to pass to the C function
            np.ascontiguousarray(grid_pos), np.ascontiguousarray(coords),
            N_coord, N_grid_points, dists
        )
        return dists.reshape(N_grid_points, N_coord)

    def gen_all_grids(
            self, grid_pos, coords, charges, epsilons, vdw_rs,
            rad_dielec_const, elec_rep_max, elec_attr_max,
            vwd_rep_max, vwd_attr_max
        ):
        """Generate the electrostatic and van der Waals grids."""
        # ensure the correct sign for the electrostatics and van der Waal potentials
        elec_rep_max = abs(elec_rep_max)
        elec_attr_max = -abs(elec_attr_max)
        vwd_rep_max = abs(vwd_rep_max)
        vwd_attr_max = -abs(vwd_attr_max)
    
        N_coord, N_grid_points = coords.shape[0], grid_pos.shape[0]
        elec_grid = np.zeros(N_grid_points, dtype=np.float32, order='C')
        vdw_grid_attr = np.zeros(N_grid_points, dtype=np.float32, order='C')
        vdw_grid_rep = np.zeros(N_grid_points, dtype=np.float32, order='C')

        self._gen_all_grids(
            # Ensure contiguity of the arrays to pass to the C function
            np.ascontiguousarray(grid_pos, dtype=np.float32), 
            np.ascontiguousarray(coords, dtype=np.float32),
            charges, epsilons, vdw_rs, CC_ELEC, rad_dielec_const,
            elec_rep_max, elec_attr_max, vwd_rep_max, vwd_attr_max,
            N_coord, N_grid_points, elec_grid, vdw_grid_attr, vdw_grid_rep
        )
        return elec_grid, vdw_grid_attr, vdw_grid_rep

# Classes for the ligand grid generation for C struct
class _Vector3d(ctypes.Structure):
    _fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float),
                ('z', ctypes.c_float)]

class _Dim3d(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int),
                ('y', ctypes.c_int),
                ('z', ctypes.c_int)]

class Grid(ctypes.Structure):
    lib = ctypes.CDLL(
        os.path.join(os.path.dirname(__file__), 'probe_grid_gen_simple.so')
    )
    _fields_ = [('dim', _Dim3d),
                ('N_grid_points', ctypes.c_int),
                ('origin', _Vector3d),
                ('spacing', ctypes.c_float),
                ('coords', ctypes.POINTER(_Vector3d)),
                ('lig_coords', ctypes.POINTER(ctypes.c_float)),
                ('elec_grid', ctypes.POINTER(ctypes.c_float)),
                ('vdw_grid_attr', ctypes.POINTER(ctypes.c_float)),
                ('vdw_grid_rep', ctypes.POINTER(ctypes.c_float))]

    def __repr__(self):
        return (
            f'<ParamGrid dim=({self.dim.x}, {self.dim.y}, {self.dim.z}) '
            f'spacing={self.spacing}>'
        )

    def __del__(self):
        self.dealloc()

    def dealloc(self):
        """Deallocate the grid."""
        self.lib.dealloc_grid(self)

class ProbeGridCompEngine:
    """Wrapper for the C++ receptor grid generation library."""
    def __init__(self):
        self.lib = ctypes.CDLL(
            os.path.join(os.path.dirname(__file__), 'probe_grid_gen_simple.so')
        )
        self._batch_quat_rotate = None
        self._rotate_gen_grids_eps_rmin = None 
        self._rotate_gen_grids = None
        self._gen_lig_grid = None
        self._dealloc_grid = None

        # Define the argument dtype and return types for the C functions
        # This is for taking the rmin and epsilon values then computing the AB coefficients
        self._rotate_gen_grids_eps_rmin = self.lib.rotate_gen_lig_grids_eps_rmin
        self._rotate_gen_grids_eps_rmin.restype = ctypes.POINTER(Grid)
        self._rotate_gen_grids_eps_rmin.argtypes = [
            ctypes.c_float, # grid_spacing
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # epsilons
            nd_float_ptr_type, # vdw_rs
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_coords
            nd_float_ptr_type, # quats
            ctypes.c_int # N_quats
        ]

        # This is for taking the AB coefficients directly
        self._rotate_gen_grids = self.lib.rotate_gen_lig_grids
        self._rotate_gen_grids.restype = ctypes.POINTER(Grid)
        self._rotate_gen_grids.argtypes = [
            ctypes.c_float, # grid_spacing
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # vdw_attr_factors (B coefficients)
            nd_float_ptr_type, # vdw_rep_factors (A coefficients)
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_coords
            nd_float_ptr_type, # quats
            ctypes.c_int # N_quats
        ]

        self._gen_lig_grid = self.lib.gen_lig_grid
        self._gen_lig_grid.restype = Grid
        self._gen_lig_grid.argtypes = [
            ctypes.c_float, # grid_spacing
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # vdw_attr_factors
            nd_float_ptr_type, # vdw_rep_factors
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_coords
        ]

        self._batch_quat_rotate = self.lib.batch_quatornion_rotate
        self._batch_quat_rotate.restype = None
        self._batch_quat_rotate.argtypes = [
            nd_float_ptr_type, # quats
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_quats
            ctypes.c_int, # N_coords
            nd_float_ptr_type # out_coords
        ]

        self._dealloc_grid = self.lib.dealloc_grid
        self._dealloc_grid.restype = None
        self._dealloc_grid.argtypes = [Grid]

    @property
    def backend(self):
        # This is a dummy property to make the API consistent with the
        # ReceptorGridCompEngine class.
        # CUDA has not implemented for the probe grid generation yet.
        return 'cpu'

    def set_backend(self, backend):
        """This is a dummy method to make the API consistent with the
        ReceptorGridCompEngine class.
        CUDA has not implemented for the probe grid generation yet."""
        if backend == 'cuda':
            warnings.warn(
                'CUDA backend requested, but CUDA not available. '
                'Use CPU backend instead.'
            )

    def batch_quat_rotate(self, quats, coords):
        """Rotate a batch of coordinates by a batch of quaternions (scalar-first)."""
        N_quats, N_coords = quats.shape[0], coords.shape[0]
        out_coords = np.zeros((N_quats, N_coords, 3), dtype=np.float32, order='C')
        self._batch_quat_rotate(
            # Ensure contiguity of the arrays to pass to the C function
            np.ascontiguousarray(quats, dtype=np.float32), 
            np.ascontiguousarray(coords, dtype=np.float32),
            N_quats, N_coords, out_coords
        )
        return out_coords

    def rotate_gen_grids_eps_rmin(
            self, grid_spacing, charges, epsilons, vdw_rs, coords, quats
        ):
        """Rotate the ligand grids by a batch of quaternions. Use the rmin and 
        epsilon values. The quaternions should be scalar-first."""
        N_quats, N_coords = quats.shape[0], coords.shape[0]
        grids = self._rotate_gen_grids_eps_rmin(
            grid_spacing, 
            np.ascontiguousarray(charges, dtype=np.float32),
            np.ascontiguousarray(epsilons, dtype=np.float32),
            np.ascontiguousarray(vdw_rs, dtype=np.float32),
            np.ascontiguousarray(coords, dtype=np.float32), N_coords,
            np.ascontiguousarray(quats, dtype=np.float32), N_quats
        )
        grids = [grids[i] for i in range(N_quats)]
        return grids

    def rotate_gen_grids(
            self, grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, 
            coords, quats
        ):
        """Rotate the ligand grids by a batch of quaternions. Use the A and B
        coefficients. The quaternions should be scalar-first."""
        N_quats, N_coords = quats.shape[0], coords.shape[0]
        grids = self._rotate_gen_grids(
            grid_spacing, 
            np.ascontiguousarray(charges, dtype=np.float32),
            np.ascontiguousarray(vdw_attr_factors, dtype=np.float32), 
            np.ascontiguousarray(vdw_rep_factors, dtype=np.float32), 
            np.ascontiguousarray(coords, dtype=np.float32), N_coords,
            np.ascontiguousarray(quats, dtype=np.float32), N_quats
        )
        grids = [grids[i] for i in range(N_quats)]
        return grids

    def gen_lig_grid(
            self, grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, coords
        ):
        """Generate the ligand grid without any rotation."""
        N_coords = coords.shape[0]
        grid = self._gen_lig_grid(
            grid_spacing, 
            np.ascontiguousarray(charges, dtype=np.float32),
            np.ascontiguousarray(vdw_attr_factors, dtype=np.float32), 
            np.ascontiguousarray(vdw_rep_factors, dtype=np.float32), 
            np.ascontiguousarray(coords, dtype=np.float32), N_coords
        )
        return grid

    def dealloc_grid(self, grid):
        """Deallocate the grid."""
        self._dealloc_grid(grid)

