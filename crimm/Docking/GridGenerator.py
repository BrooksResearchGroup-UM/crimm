import os
import numpy as np
# C extension for FFT docking related routines
from crimm import fft_docking 
from crimm.Visualization import View
from crimm.Modeller import ParameterLoader
from crimm.Docking.GridShapes import (
    _Grid, CubeGrid, BoundingBoxGrid, TruncatedSphereGrid, ConvexHullGrid
)
from crimm.Data.constants import CC_ELEC_CHARMM as CC_ELEC

data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../Data')
)

class GridCoordGenerator:
    def __init__(self, grid_spacing, padding, optimize_for_fft) -> None:
        self.spacing = grid_spacing
        self.paddings = padding
        self.entity = None
        self.coords = None
        self.optimize_for_fft = optimize_for_fft
        self._cubic_grid = None
        self._bounding_box_grid = None
        self._truncated_sphere_grid = None
        self._enlarged_convex_hull_grid = None
        # the current grid used for energy calculations
        self.coord_grid : _Grid = None 
        self._elec_grid = None
        self._vdw_grid_attr = None
        self._vdw_grid_rep = None
        self._dists = None
        self._charges = None
        self._epsilons = None
        self._vdw_rs = None
        self._grid_shape = None
        self.param_loader = {}

    def load_entity(self, entity):
        """Load entity and set grid spacing and paddings."""
        self.entity = entity
        self.coords = self._extract_coords()
        # remove all grid attributes if existing
        self._cubic_grid = None
        self._bounding_box_grid = None
        self._truncated_sphere_grid = None
        self._enlarged_convex_hull_grid = None

    def _extract_coords(self) -> np.array:
        coords = []
        for atom in self.entity.get_atoms():
            coords.append(atom.coord)
        return np.asarray(coords, dtype=np.float32)

    @property
    def coord_center(self):
        """Return the center of the coordinates (N, 3) of the loaded entity. 
        The center is defined as the midpoint of the maximum and minimum 
        coordinates of each dimension. Should return (0, 0, 0) after the 
        transformation by `CoordManipulator.orient_coords()`.
        """
        return (self.coords.max(0) + self.coords.min(0))/2

    @property
    def max_dims(self):
        """Return the dimensions of the bounding box (in angstrom) of the 
        entity coordinates (x, y, z).
        """
        if self.coords is not None:
            return self.coords.ptp(0)

    @property
    def cubic_grid(self):
        """Return a grid of points (N, 3) that covers the bounding cube of the coordinates."""
        if self._cubic_grid is None:
            self._cubic_grid = CubeGrid(
                self.max_dims, self.coord_center, self.spacing, self.paddings, 
                self.optimize_for_fft
            )
        return self._cubic_grid

    @property
    def bounding_box_grid(self):
        """Return a grid of points (N, 3) that covers the bounding box of the coordinates."""
        if self._bounding_box_grid is None:
            self._bounding_box_grid = BoundingBoxGrid(
                self.max_dims, self.coord_center, self.spacing, self.paddings,
                self.optimize_for_fft
            )
        return self._bounding_box_grid

    @property
    def truncated_sphere_grid(self):
        """Return a grid of points (N, 3) that covers the truncated sphere of the coordinates."""
        if self._truncated_sphere_grid is None:
            self._truncated_sphere_grid = TruncatedSphereGrid(
                self.max_dims, self.coord_center, self.spacing, self.paddings,
                self.optimize_for_fft
            )
        return self._truncated_sphere_grid

    @property
    def convex_hull_grid(self):
        """Return a grid of points (N, 3) that covers the convex hull of the coordinates."""
        if self._enlarged_convex_hull_grid is None:
            self._enlarged_convex_hull_grid = ConvexHullGrid(
                self.coords, self.max_dims,
                self.coord_center, self.spacing, self.paddings, self.optimize_for_fft
            )
        return self._enlarged_convex_hull_grid

    def show_hull_surface(self, show_licorice=False, show_enlarged_hull=False):
        """Show the surface of the convex hull."""
        q_hull = self.convex_hull_grid.Qhull
        if show_enlarged_hull:
            vertices = self.convex_hull_grid.enlarged_hull_vertices
            idx_dict = {x: idx for idx, x in enumerate(q_hull.vertices)}
            enlarged_simplex_ids = np.vectorize(idx_dict.get)(q_hull.simplices)
            flattened_array = (
                vertices[enlarged_simplex_ids].reshape(-1)
            )
        else:
            flattened_array = q_hull.points[q_hull.simplices].reshape(-1)
        view = View()
        view.load_entity(self.entity)
        if show_licorice:
            view.clear_representations()
            view.add_representation('licorice', selection='protein')

        hull_shape = view.shape.add(
            'mesh',
            flattened_array,
            np.ones_like(flattened_array)*0.7,
        )

        hull_shape.add_surface(opacity=0.2)

        return view

    def _collect_params(self):
        charges = []
        vdw_rs = []
        epsilons = []
        for atom in self.entity.get_atoms():
            atom_type = atom.topo_definition.atom_type
            charges.append(atom.topo_definition.charge)
            nb_param = self.param_loader['nonbonded'][atom_type]
            vdw_rs.append(nb_param.rmin_half) 
            epsilons.append(nb_param.epsilon)
        # Single precision is enough for energy calculations
        self._charges = np.asarray(charges, dtype=np.float32, order='C')
        self._vdw_rs = np.asarray(vdw_rs, dtype=np.float32, order='C')
        self._vdw_rs *= 2 # convert rmin_half to rmin
        self._epsilons = np.asarray(epsilons, dtype=np.float32, order='C')

    def _fill_dx(self, grid, values_str, spacing=None):
        xd, yd, zd = self.coord_grid.points_per_dim
        min_x, min_y, min_z = self.coord_grid.min_coords
        if spacing is None:
            spacing = self.coord_grid.spacing
        dx_template = (
            f'''#Generated dx file for fft grid
object 1 class gridpositions counts {xd} {yd} {zd}
origin {min_x:e} {min_y:e} {min_z:e}
delta {spacing:e} 0.000000e+000 0.000000e+000
delta 0.000000e+000 {spacing:e} 0.000000e+000
delta 0.000000e+000 0.000000e+000 {spacing:e}
object 2 class gridconnections counts {xd} {yd} {zd}
object 3 class array type double rank 0 items {grid.size} data follows
{values_str}
attribute "dep" string "positions"
object "regular positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3'''
        )
        return dx_template

class ReceptorGridGenerator(GridCoordGenerator):
    """A potential energy grid generator for a receptor molecule."""
    grid_shape_dict = {
        'cubic': 'cubic_grid',
        'bounding_box': 'bounding_box_grid',
        'truncated_sphere': 'truncated_sphere_grid',
        'convex_hull' : 'convex_hull_grid'
    }
    def __init__(
            self, grid_spacing, padding, optimize_for_fft,
            rad_dielec_const=2.0, elec_rep_max=40, elec_attr_max=-20,
            vdw_rep_max=2.0, vdw_attr_max=-1.0, use_constant_dielectric=False
        ) -> None:
        super().__init__(grid_spacing, padding, optimize_for_fft)
        self.optimize_for_fft = optimize_for_fft
        self.rad_dielec_const = abs(rad_dielec_const)
        self.elec_rep_max = abs(elec_rep_max)
        self.elec_attr_max = -abs(elec_attr_max)
        self.vdw_rep_max = abs(vdw_rep_max)
        self.vdw_attr_max = -abs(vdw_attr_max)
        self.use_cdie = use_constant_dielectric
        self.potential_grids = None

    def load_entity(self, entity, grid_shape='convex_hull'):
        """Load an entity and generate the grid coordinates and energy potentials
        associated with each grid point in space.
        
        Parameters
        ----------
        entity : :obj:`crimm.StructEntity.Chain` 
        The entity to be loaded.
        The spacing of the grid in Angstroms.
        padding : float
        The padding to be added to the grid dimensions (in Angstroms).
        grid_shape : str, optional
        The geometric shape of the grid. Must be one of 'cubic', 'bounding_box', 
        'truncated_sphere', or 'convex_hull'. Default is 'convex_hull'.
        """
        if grid_shape not in self.grid_shape_dict:
            raise ValueError(
                f'grid_type must be one of {list(self.grid_shape_dict.keys())}'
            )
        #TODO: add support for multiple chains
        if entity.level != 'C':
            raise ValueError(
                f'entity must be a chain, got {entity.level}'
            )
        if entity.chain_type == "Polypeptide(L)":
            self.param_loader = ParameterLoader('protein')
        elif entity.chain_type == "Polyribonucleotide":
            self.param_loader = ParameterLoader('nucleic')
        else:
            raise ValueError(
                f'entity must be a protein or an RNA, got {self.entity.chain_type}'
            )
        super().load_entity(entity)
        self._grid_shape = grid_shape
        self.coord_grid = getattr(self, self.grid_shape_dict[grid_shape])
        self._collect_params()
        # clear the pairwise dists and grids
        self._dists = None
        self._elec_grid = None
        self._vdw_grid_attr = None
        self._vdw_grid_rep = None

    def get_coord_grid(self):
        """Return the coordinate grid used for the energy calculations."""
        if self.coord_grid is None:
            raise ValueError(
                'No grid is loaded. Please load an entity first.'
            )
        return self.coord_grid

    def get_pairwise_dists(self):
        grid = self.get_coord_grid()
        if self._dists is None:
            self._dists = fft_docking.pairwise_dist(grid.coords, self.coords)
        return self._dists

    def get_attr_vdw_grid(self, convert_shape=True):
        """Get the van der Waals attractive energy grid. If convert_shape is True,
        the grid will be converted to the bounding box shape, where the void will
        be filled with zeros. The returned grid will be 3D array. Otherwise, the
        grid will be returned as a 1D array with only values within the shape."""
        if self._vdw_grid_attr is None:
            self.genernate_grids()
        if convert_shape:
            return self.convert_to_3d_grid(self._vdw_grid_attr)
        return self._vdw_grid_attr

    def get_rep_vdw_grid(self, convert_shape=True):
        """Get the van der Waals repulsive energy grid. If convert_shape is True,
        the grid will be converted to the bounding box shape, where the void will
        be filled with zeros. The returned grid will be 3D array. Otherwise, the
        grid will be returned as a 1D array with only values within the shape."""
        if self._vdw_grid_rep is None:
            self.genernate_grids()
        if convert_shape:
            return self.convert_to_3d_grid(self._vdw_grid_rep)
        return self._vdw_grid_rep

    def get_elec_grid(self, convert_shape=True):
        """Get the electrostatic energy grid. If convert_shape is True,
        the grid will be converted to the bounding box shape, where the void will
        be filled with zeros. The returned grid will be 3D array. Otherwise, the
        grid will be returned as a 1D array with only values within the shape."""
        if self._elec_grid is None:
            self.genernate_grids()
        if convert_shape:
            return self.convert_to_3d_grid(self._elec_grid)
        return self._elec_grid

    def get_potential_grids(self):
        """Get all potential energy grids (electrostatic, van der Waals attractive, and
        van der Waals repulsive). The returned grids will be 3D array."""
        if self.potential_grids is None:
            self.genernate_grids()
        return self.potential_grids

    def genernate_grids(self):
        """Generate and return all grids (electrostatic, van der Waals attractive, and
        van der Waals repulsive) as a 3D array."""
        self._elec_grid, self._vdw_grid_attr, self._vdw_grid_rep = \
            fft_docking.generate_grids(
                self.coord_grid.coords, self.coords, self._charges, self._epsilons,
                self._vdw_rs, CC_ELEC, self.rad_dielec_const, self.elec_rep_max,
                self.elec_attr_max, self.vdw_rep_max, self.vdw_attr_max, self.use_cdie
            )

        self.potential_grids = np.stack((
            self.convert_to_3d_grid(self._elec_grid),
            self.convert_to_3d_grid(self._vdw_grid_attr),
            self.convert_to_3d_grid(self._vdw_grid_rep)
        )).astype(np.float32)

    def convert_to_boxed_grid(self, grid):
        """Fill a non-box-shaped grid to bounding box shape, where the void will 
        be filled with zeros. The returned grid will be 1D array."""
        if self._grid_shape in ('convex_hull', 'truncated_sphere'):
            # Place the grid values back in the correct positions in the 
            # bounding box grid
            boxed_grid = np.zeros(self.bounding_box_grid.coords.shape[0])
            fill_ids = self.coord_grid.grid_ids_in_box
            boxed_grid[fill_ids] = grid
        else:
            boxed_grid = grid
        return boxed_grid

    def convert_to_3d_grid(self, grid_vals):
        """Convert a 1D grid array to a 3D grid array. Values in trucated sphere 
        and convex hull grids will be converted to the bounding box grid, 
        and their void will be filled with zeros."""
        boxed_grid = self.convert_to_boxed_grid(grid_vals)
        return boxed_grid.reshape(self.coord_grid.points_per_dim)

    def save_dx(self, filename, grid_vals, convert_shape=True):
        """Save a grid to a .dx file."""
        if convert_shape:
            boxed_grid = self.convert_to_boxed_grid(grid_vals)
        else:
            # Assume the grid is already in the correct box shape
            boxed_grid = grid_vals
        
        values_str = ''
        counter = 0
        for value in boxed_grid:
            counter += 1
            values_str += f'{value:e} '
            if counter % 6 == 0:
                values_str += '\n'

        dx_str = self._fill_dx(boxed_grid, values_str, self.spacing)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(dx_str)
            f.flush() # flush the buffer to ensure the file is written

class ProbeGridGenerator(GridCoordGenerator):
    """A potential energy grid generator for a small molecule probe.
    
    Parameters
    ----------
    grid_spacing : float
    The spacing of the grid in Angstroms.
    rotation_search_level : int, optional
    The level of rotations to be used for the probe. Must be one of 0, 1, 2, or 3.
    Default is 2. Number of rotations : {0: No rotation, 1: 576, 2: 4068, 3: 36864}
    The rotations are represented by quaternions (scalar-first) that are evenly 
    sampled from the rotational space.
    """
    _rot_search_levels = {
        0: np.array([[1, 0, 0, 0]], dtype=np.float32), # identity quaternion (no rotation)
        1: np.load(os.path.join(data_dir, 'quaternion-1.npy')).astype(np.float32),
        2: np.load(os.path.join(data_dir, 'quaternion-2.npy')).astype(np.float32),
        3: np.load(os.path.join(data_dir, 'quaternion-3.npy')).astype(np.float32)
    }

    def __init__(self, grid_spacing, rotation_search_level=2) -> None:
        super().__init__(grid_spacing, padding=0, optimize_for_fft=False)
        if rotation_search_level not in self._rot_search_levels:
            raise ValueError(
                f'rotation_search_level must be one of {list(self._rot_search_levels.keys())}'
            )
        self._grid_shape = "bounding_box"
        self.param_loader = ParameterLoader('cgenff')
        self.param_grids = None
        self._elec_grid = None
        self._vdw_grid_attr = None
        self._vdw_grid_rep = None
        self.quats = self._rot_search_levels[rotation_search_level]
        self.rotated_coords = None

    def load_probe(self, probe):
        """Load an entity and generate the grid coordinates and energy potentials
        associated with each grid point in space.
        
        Parameters
        ----------
        entity : :obj:`crimm.Data.probes._Probe` 
        The entity to be loaded.
        """
        super().load_entity(probe)
        self._grid_shape = "cubic"
        self._collect_params()
        self.param_grids = None
        self._elec_grid = None
        self._vdw_grid_attr = None
        self._vdw_grid_rep = None
        self.rotated_coords = None
        self._vdw_attr_factor, self._vdw_rep_factor =\
        fft_docking.calc_vdw_energy_factors(self._epsilons, self._vdw_rs)

    def generate_grids(self, quats=None):
        """Generate the electrostatic and van der Waals energy grids for a probe.
        Quaternion rotations will be applied first to obtain various orientations.
        The grids will be generated for all orientations. 
        
        Parameters
        ----------
        quats : np.array 
        A (N, 4) array of quaternions (scalar-first) to apply to the probe. Default to None, 
        and the default quaternions set based on selected the rotation level will be used.

        Returns
        -------
        self.param_grids : np.array
        A 5D array of grids (3, N, x, y, z) where N is the number of orientations, and
        (x, y, z) are the grid dimensions. For the first dimension (3), the first array is the
        **electrostatic** grid, the second array is the **van der Waals attractive**
        grid, and the third array is the **van der Waals repulsive** grid.
        i.e. elec_grid, vdw_attr_grid, vdw_rep_grid = grids
        """
        if self._epsilons is None:
            raise ValueError(
                'No probe is loaded. Please load a probe first.'
            )

        if quats is None:
            quats = self.quats
        self.rotated_coords, self._elec_grid, self._vdw_attr_grid, self._vdw_rep_grid =\
        fft_docking.rotate_gen_lig_grids(
            self.spacing, self._charges,
            self._vdw_attr_factor, self._vdw_rep_factor,
            self.coords, self.quats
        )
        N_quats, grid_dim, _, _ = self._elec_grid.shape
        self.param_grids = np.empty(
            (N_quats, 3, grid_dim, grid_dim, grid_dim), dtype=np.float32
        )
        self.param_grids[:, 0] = self._elec_grid
        self.param_grids[:, 1] = self._vdw_attr_grid
        self.param_grids[:, 2] = self._vdw_rep_grid
    
    def generate_grids_single_pose(self, pose_coords):
        """Generate the electrostatic and van der Waals energy grids for a single pose.
        Quaternion rotations will be applied first to obtain various orientations.
        The grids will be generated for all orientations. 
        
        Parameters
        ----------
        pose_coords : np.array
        A (N, 3) array of coordinates of the pose.
        
        Returns
        -------
        grids : np.array
        A 5D array of grids (3, x, y, z), where (x, y, z) are the grid dimensions.
        For the first dimension (3), the first array is the **electrostatic** grid, 
        the second array is the **van der Waals attractive** grid, 
        and the third array is the **van der Waals repulsive** grid.
        i.e. elec_grid, vdw_attr_grid, vdw_rep_grid = grids
        """
        if self._epsilons is None:
            raise ValueError(
                'No probe is loaded. Please load a probe first.'
            )

        if pose_coords.shape != self.coords.shape:
            raise ValueError(
                'The number of coordinates and number of atoms mismatch!'
            )

        _, elec_grid, vdw_attr_grid, vdw_rep_grid =\
        fft_docking.rotate_gen_lig_grids(
            self.spacing, self._charges,
            self._vdw_attr_factor, self._vdw_rep_factor,
            pose_coords, self._rot_search_levels[0]
        )
        grids = np.squeeze(np.stack(
            (elec_grid, vdw_attr_grid, vdw_rep_grid), dtype=np.float32
        ))
        return grids

    def get_param_grids(self, quats=None):
        """Return the generated grids for the probe."""
        if self.param_grids is None:
            self.generate_grids(quats=quats)
        return self.param_grids
    
    def get_roated_coords(self):
        """Return the rotated coordinates of the probe."""
        if self.rotated_coords is None:
            raise ValueError(
                'No rotated coordinates are available. Please generate the grids first.'
            )
        return self.rotated_coords

    def save_dx(self, filename, grid_vals):
        """Save a grid to a .dx file."""
        values_str = ''
        counter = 0
        for value in grid_vals:
            counter += 1
            values_str += f'{value:e} '
            if counter % 6 == 0:
                values_str += '\n'

        dx_str = self._fill_dx(grid_vals, values_str, spacing=self.spacing)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(dx_str)
