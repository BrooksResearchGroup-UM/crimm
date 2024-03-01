import os
import warnings
import ctypes
import numpy as np
from scipy.spatial.distance import pdist
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
                ('N_lig_coords', ctypes.c_int),
                ('_origin', _Vector3d),
                ('spacing', ctypes.c_float),
                ('_coords', ctypes.POINTER(_Vector3d)),
                ('_lig_coords', ctypes.POINTER(ctypes.c_float)),
                ('_elec_grid', ctypes.POINTER(ctypes.c_float)),
                ('_vdw_grid_attr', ctypes.POINTER(ctypes.c_float)),
                ('_vdw_grid_rep', ctypes.POINTER(ctypes.c_float))]

    @property
    def shape(self):
        """Get the grid dimension."""
        return (self.dim.x, self.dim.y, self.dim.z)

    @property
    def origin(self):
        """Get the grid origin."""
        return np.array((self._origin.x, self._origin.y, self._origin.z))
    
    @property
    def lig_coords(self):
        """Get the ligand coordinates."""
        return np.ctypeslib.as_array(self._lig_coords, (self.N_lig_coords, 3))

    @property
    def elec_grid(self):
        """Get the electrostatic grid."""
        return np.ctypeslib.as_array(self._elec_grid, self.shape)

    @property
    def attr_vdw_grid(self):
        """Get the van der Waals attractive grid."""
        return np.ctypeslib.as_array(self._vdw_grid_attr, self.shape)

    @property
    def rep_vdw_grid(self):
        """Get the van der Waals repulsive grid."""
        return np.ctypeslib.as_array(self._vdw_grid_rep, self.shape)

    @property
    def max_coord(self):
        """Get the coordinates of the grid corners."""
        coord = self._coords[self.N_grid_points-1]
        return np.array((coord.x, coord.y, coord.z))

    @property
    def min_coord(self):
        """Get the coordinates of the grid corners."""
        coord = self._coords[0]
        return np.array((coord.x, coord.y, coord.z))

    def __repr__(self):
        return (
            f'<ParamGrid shape={self.shape} spacing={self.spacing}>'
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
            os.path.join(os.path.dirname(__file__), 'lig_grid_gen.so')
        )
        # Define the argument dtype and return types for the C functions
        # This is for taking the AB coefficients
        self._rotate_gen_grids = self.lib.rotate_gen_lig_grids
        self._rotate_gen_grids.restype = None
        self._rotate_gen_grids.argtypes = [
            ctypes.c_float, # grid_spacing
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # vdw_attr_factors
            nd_float_ptr_type, # vdw_rep_factors
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_coords
            nd_float_ptr_type, # quats
            ctypes.c_int, # N_quats
            ctypes.c_int, # cube_dim
            nd_float_ptr_type, # rot_coords
            nd_float_ptr_type, # elec_grids
            nd_float_ptr_type, # vdw_grids_attr
            nd_float_ptr_type, # vdw_grids_rep
        ]

        # calculate the AB coefficients from epsilon and r_min
        self._calc_vdw_energy_factors = self.lib.calc_vdw_energy_factors
        self._calc_vdw_energy_factors.restype = None
        self._calc_vdw_energy_factors.argtypes = [
            nd_float_ptr_type, # epsilons
            nd_float_ptr_type, # vdw_rs
            ctypes.c_int, # N_coords
            nd_float_ptr_type, # vdw_attr_factor
            nd_float_ptr_type, # vdw_rep_factor
        ]

        self._gen_lig_grid = self.lib.gen_lig_grid
        self._gen_lig_grid.restype = None
        self._gen_lig_grid.argtypes = [
            ctypes.c_float, # grid_spacing
            nd_float_ptr_type, # charges
            nd_float_ptr_type, # vdw_attr_factors
            nd_float_ptr_type, # vdw_rep_factors
            nd_float_ptr_type, # coords
            ctypes.c_int, # N_coords
            ctypes.c_int, # cube_dim
            nd_float_ptr_type, # elec_grid
            nd_float_ptr_type, # vdw_grid_attr
            nd_float_ptr_type # vdw_grid_rep
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

    def calc_vdw_energy_factors(self, epsilons, vdw_rs):
        """Calculate the van der Waals energy factors."""
        N_coords = epsilons.shape[0]
        vdw_attr_factor = np.zeros(N_coords, dtype=np.float32, order='C')
        vdw_rep_factor = np.zeros(N_coords, dtype=np.float32, order='C')
        self._calc_vdw_energy_factors(
            # Ensure contiguity of the arrays to pass to the C function
            np.ascontiguousarray(epsilons, dtype=np.float32), 
            np.ascontiguousarray(vdw_rs, dtype=np.float32),
            N_coords, vdw_attr_factor, vdw_rep_factor
        )
        return vdw_attr_factor, vdw_rep_factor

    def rotate_gen_grids(
            self, grid_spacing, charges, vdw_attr_factor, vdw_rep_factor, coords, quats
        ):
        """Rotate the ligand grids by a batch of quaternions. Use the rmin and 
        epsilon values. The quaternions should be scalar-first."""
        N_quats, N_coords = quats.shape[0], coords.shape[0]
        max_dist = pdist(coords).max()
        cube_dim = np.ceil(max_dist/grid_spacing).astype(int)

        rot_coords = np.zeros((N_quats, N_coords, 3), dtype=np.float32)
        elec_grids = np.zeros((N_quats, cube_dim, cube_dim, cube_dim), dtype=np.float32)
        vdw_grids_attr = np.zeros_like(elec_grids, dtype=np.float32)
        vdw_grids_rep = np.zeros_like(elec_grids, dtype=np.float32)
        self._rotate_gen_grids(
            grid_spacing,
            np.ascontiguousarray(charges, dtype=np.float32),
            np.ascontiguousarray(vdw_attr_factor, dtype=np.float32),
            np.ascontiguousarray(vdw_rep_factor, dtype=np.float32),
            np.ascontiguousarray(coords, dtype=np.float32), 
            N_coords,
            np.ascontiguousarray(quats, dtype=np.float32), 
            N_quats,
            cube_dim,
            rot_coords,
            elec_grids,
            vdw_grids_attr,
            vdw_grids_rep
        )
        all_grids = np.empty(
            (N_quats, 3, cube_dim, cube_dim, cube_dim), dtype=np.float32
        )
        all_grids[:, 0] = elec_grids
        all_grids[:, 1] = vdw_grids_attr
        all_grids[:, 2] = vdw_grids_rep
        return all_grids, rot_coords

    def gen_lig_grid(
            self, grid_spacing, charges, vdw_attr_factors, vdw_rep_factors, coords,
            cube_dim
        ):
        """Generate the ligand grid without any rotation."""
        N_coords = coords.shape[0]
        min_corner = coords.min(axis=0)
        min_corner = _Vector3d(*min_corner)
        elec_grids = np.zeros((cube_dim, cube_dim, cube_dim), dtype=np.float32)
        vdw_grids_attr = np.zeros_like(elec_grids, dtype=np.float32)
        vdw_grids_rep = np.zeros_like(elec_grids, dtype=np.float32)
        self._gen_lig_grid(
            grid_spacing, 
            np.ascontiguousarray(charges, dtype=np.float32),
            np.ascontiguousarray(vdw_attr_factors, dtype=np.float32), 
            np.ascontiguousarray(vdw_rep_factors, dtype=np.float32), 
            np.ascontiguousarray(coords, dtype=np.float32), 
            N_coords,
            min_corner,
            cube_dim,
            elec_grids,
            vdw_grids_attr,
            vdw_grids_rep
        )
        all_grids = np.array([elec_grids, vdw_grids_attr, vdw_grids_rep])
        return all_grids

