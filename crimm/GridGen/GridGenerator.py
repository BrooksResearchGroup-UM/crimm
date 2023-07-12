from itertools import product
import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull, Delaunay
from crimm.Visualization import View
from crimm.Modeller import ParameterLoader
from crimm.GridGen._grid_gen_wrappers import GridCompEngine

class GridCoordGenerator:
    def __init__(self, grid_spacing, padding) -> None:
        self.spacing = grid_spacing
        self.paddings = padding
        self.entity = None
        self.coords = None
        self._cubic_grid = None
        self._bounding_box_grid = None
        self._truncated_sphere_grid = None
        self._enlarged_convex_hull_grid = None

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
        return np.asarray(coords)

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
                self.max_dims, self.coord_center, self.spacing, self.paddings
            )
        return self._cubic_grid

    @property
    def bounding_box_grid(self):
        """Return a grid of points (N, 3) that covers the bounding box of the coordinates."""
        if self._bounding_box_grid is None:
            self._bounding_box_grid = BoundingBoxGrid(
                self.max_dims, self.coord_center, self.spacing, self.paddings
            )
        return self._bounding_box_grid

    @property
    def truncated_sphere_grid(self):
        """Return a grid of points (N, 3) that covers the truncated sphere of the coordinates."""
        if self._truncated_sphere_grid is None:
            self._truncated_sphere_grid = TruncatedSphereGrid(
                self.max_dims, self.coord_center, self.spacing, self.paddings
            )
        return self._truncated_sphere_grid

    @property
    def convex_hull_grid(self):
        """Return a grid of points (N, 3) that covers the convex hull of the coordinates."""
        if self._enlarged_convex_hull_grid is None:
            self._enlarged_convex_hull_grid = ConvexHullGrid(
                self.coords, self.max_dims,
                self.coord_center, self.spacing, self.paddings
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

class _Grid:
    def __init__(self, coord_center, spacing, padding) -> None:
        self._center = coord_center
        self._spacing = spacing
        self._padding = padding
        self._coords = None
        self._points_per_dim = None
        self.min_coords = None

    @property
    def box_lengths(self):
        """Return the dimensions of the bounding box (in angstrom) of the 
        grid coordinates (x, y, z).
        """
        if self._coords is not None:
            return self._coords.ptp(0)

    @property
    def points_per_dim(self):
        """Return the number of grid points on each dimension of the 
        bounding box grid."""
        return self._points_per_dim

    @property
    def coords(self):
        """Return the coordinates of the grid points (N, 3) """
        return self._coords

    @staticmethod
    def _get_box_grid(center, grid_half_widths, spacing):
        """Return a grid of points (N, 3) defined by a center (x, y, z) and the
        grid box's half widths (x_len/2, y_len/2, z_len/2)."""
        dims = []
        for mid_point, half_width in zip(center, grid_half_widths):
            dims.append(
                np.arange(
                    mid_point-half_width,
                    mid_point+half_width+spacing,
                    spacing
                )
            )
        grid_pos = np.array(
            np.meshgrid(*dims, indexing='ij')
        ).reshape(3,-1).T
        x_pos, y_pos, z_pos = dims
        dim_sizes = np.array([x_pos.size, y_pos.size, z_pos.size])
        return dim_sizes, grid_pos

class CubeGrid(_Grid):
    """A grid of points (N, 3) that covers the bounding cube of the coordinates."""
    def __init__(self, dims, coord_center, spacing, padding) -> None:
        super().__init__(coord_center, spacing, padding)
        grid_half_widths = (np.ceil(np.max(dims)/2)+padding)*np.ones(3)
        self._points_per_dim, self._coords = self._get_box_grid(
            coord_center, grid_half_widths, spacing
        )
        self.min_coords = self._coords.min(0)

class BoundingBoxGrid(_Grid):
    """A grid of points (N, 3) that covers the bounding box of the coordinates."""
    def __init__(self, dims, coord_center, spacing, padding) -> None:
        super().__init__(coord_center, spacing, padding)
        grid_half_widths = np.ceil(dims/2)+self._padding
        self._points_per_dim, self._coords = self._get_box_grid(
            coord_center, grid_half_widths, spacing
        )
        self.min_coords = self._coords.min(0)

class TruncatedSphereGrid(BoundingBoxGrid):
    """A grid of points (N, 3) that covers the truncated sphere of the coordinates."""
    def __init__(self, dims, coord_center, spacing, padding) -> None:
        super().__init__(dims, coord_center, spacing, padding)
        tolerance = spacing*1e-2
        self.bounding_box_coords = self._coords
        radius = max(self.box_lengths)/2
        # Euclidean distance normalized by semi-axes
        distances = np.linalg.norm(
            (self.bounding_box_coords-coord_center) / radius, axis=1
        )
        # the indices of the grid points within the truncated sphere
        self.grid_ids_in_box = (distances <= 1)
        self._coords = self.bounding_box_coords[self.grid_ids_in_box]
        self._shell_surface_coords = self.bounding_box_coords[
            np.abs(distances - 1) <= tolerance
        ]

    @property
    def shell_surface_coords(self):
        """Return the coordinate (N, 3) of the shell truncated sphere with paddings."""
        return self._shell_surface_coords

class ConvexHullGrid(BoundingBoxGrid):
    """A grid of points (N, 3) that covers the convex hull of the coordinates."""
    def __init__(self, entity_coords, dims, coord_center, spacing, padding) -> None:
        super().__init__(dims, coord_center, spacing, padding)
        self.bounding_box_coords = self._coords
        self.Qhull = ConvexHull(entity_coords)
        self.enlarged_hull_vertices = self._enlarge_convex_hull(entity_coords)
        # Find the Delaunay triangulation of the convex hull
        self.delaunay = Delaunay(self.enlarged_hull_vertices)
        # the indices of the grid points within the convex hull
        self.grid_ids_in_box = np.argwhere(
            self.delaunay.find_simplex(self.bounding_box_coords) >= 0
        ).reshape(-1)
        self._coords = self.bounding_box_coords[self.grid_ids_in_box]

        self._simplices_normals = None
        self._verts_i = None
        self._hull_surf_coords = None

    def _enlarge_convex_hull(self, entity_coords):
        """Return the vertices of the enlarged convex hull of the coordinates"""
        # Compute the convex hull if not already computed
        hull = self.Qhull
        # Find the centroid of the convex hull
        centroid = np.mean(entity_coords[hull.vertices], axis=0)
        # Compute the vectors from the centroid to each vertex
        vectors = entity_coords[hull.vertices] - centroid
        # Normalize the vectors to unit length
        norms = np.linalg.norm(vectors, axis=1)
        normalized_vectors = vectors / norms[:, np.newaxis]
        # Compute the displacement vector for each vertex
        displacement = normalized_vectors * self._padding
        # Enlarge the convex hull by adding the displacement vector to each vertex
        enlarged_hull_vertices = entity_coords[hull.vertices] + displacement
        return enlarged_hull_vertices

    def _find_hull_simplex_normals(self):
        hull = self.Qhull
        simplices_coords = hull.points[hull.simplices]
        normals = np.cross(
            simplices_coords[:, 0]-simplices_coords[:, 2],
            simplices_coords[:, 1]-simplices_coords[:, 2],
            axis=1
        )
        # convert to unit normals
        normals = (normals.T/norm(normals, axis=1)).T
        self._simplices_normals = normals
        self._verts_i = simplices_coords[:, 0]

        return normals

    def find_coords_to_simplices_dists(self, coords):
        if self._simplices_normals is None:
            self._find_hull_simplex_normals()
        n_verts = self._verts_i.shape[0]
        n_coords = coords.shape[0]
        verts_expanded = np.repeat(
            self._verts_i, n_coords, axis=1
        ).reshape(*self._verts_i.shape, n_coords)

        coords_expanded = np.repeat(
            coords, n_verts, axis=1
        ).reshape(*coords.shape, n_verts).T

        coords_to_verts_i = coords_expanded - verts_expanded
        coords_to_verts_i = np.einsum('ijk->ikj', coords_to_verts_i)
        # dot_product has the shape (N_simplex, N_coords)
        dist_coords_to_simplices = np.einsum(
            'ik,ijk->ij', self._simplices_normals, coords_to_verts_i
        )

        return dist_coords_to_simplices

    def get_hull_surface_coords(self):
        if self._hull_surf_coords is not None:
            return self._hull_surf_coords

        tolerance = self._spacing*1e-2
        dists = self.find_coords_to_simplices_dists(self._coords)
        surf_ids = np.argwhere(
            np.any(np.abs(dists)<tolerance, axis=0)
        ).reshape(-1)
        self._hull_surf_coords = self._coords[surf_ids]
        return self._hull_surf_coords


class EnerGridGenerator(GridCoordGenerator):
    _allowed_backend = ['cpu', 'cuda']
    grid_shape_dict = {
        'cubic': 'cubic_grid',
        'bounding_box': 'bounding_box_grid',
        'truncated_sphere': 'truncated_sphere_grid',
        'convex_hull' : 'convex_hull_grid'
    }
    comp_engine = GridCompEngine()

    def __init__(self, grid_spacing, padding) -> None:
        super().__init__(grid_spacing, padding)
        self.coord_grid : _Grid = None # the current grid used for energy calculations
        self._elec_grid = None
        self._vdw_grid = None
        self._dists = None
        self._charges = None
        self._epsilons = None
        self._vdw_rs = None
        self._grid_shape = None
        self.param_loader = None

    @property
    def backend(self):
        return self.comp_engine.backend

    @backend.setter
    def backend(self, backend):
        return self.comp_engine.set_backend(backend)

    def load_entity(self, entity, grid_shape = 'bounding_box'):
        """Load an entity and generate the grid coordinates and energy potentials
        associated with each grid point in space.
        
        Parameters
        ----------
        entity : :obj:`crimm.StructEntity.Chain` 
        The entity to be loaded.
        grid_spacing : float
        The spacing of the grid in Angstroms.
        padding : float
        The padding to be added to the grid dimensions (in Angstroms).
        grid_shape : str, optional
        The geometric shape of the grid. Must be one of 'cubic', 'bounding_box', 
        'truncated_sphere', or 'convex_hull'. Default is 'convex_hull'.
        """
        if isinstance(self, ProbeGridGenerator):
            super().load_entity(entity)
            return

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
        self.coord_grid= getattr(self, self.grid_shape_dict[grid_shape])
        self._collect_params()
        # clear the pairwise dists and grids
        self._dists = None
        self._elec_grid = None
        self._vdw_grid = None

    def _collect_params(self):
        charges = []
        vdw_rs = []
        epsilons = []
        for atom in self.entity.get_atoms():
            atom_type = atom._topo_def.atom_type
            charges.append(atom._topo_def.charge)
            nb_param = self.param_loader['nonbonded'][atom_type]
            vdw_rs. append(nb_param.rmin_half)
            epsilons.append(nb_param.epsilon)

        self._charges = np.array(charges)
        self._vdw_rs = np.array(vdw_rs)
        self._epsilons = np.array(epsilons)

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
            self._dists = self.comp_engine.cdist(
                grid.coords, self.coords
            )
        return self._dists

    def get_vdw_grid(self, probe_radius, vwd_softcore_max):
        """Get the van der Waals energy grid."""
        pairwise_dists = self.get_pairwise_dists()
        if self._vdw_grid is None:
            self._vdw_grid = self.comp_engine.gen_vdw_grid(
                pairwise_dists, self._epsilons, self._vdw_rs, 
                probe_radius, vwd_softcore_max
            )
        return self._vdw_grid

    def get_elec_grid(self, rad_dielec_const, elec_rep_max, elec_attr_max):
        """Get the electrostatic energy grid."""
        pairwise_dists = self.get_pairwise_dists()
        if self._elec_grid is None:
            self._elec_grid = self.comp_engine.gen_elec_grid(
                pairwise_dists, self._charges, rad_dielec_const,
                elec_rep_max, elec_attr_max
            )
        return self._elec_grid

    def gen_all_grids(
            self, rad_dielec_const, elec_rep_max, elec_attr_max, probe_radius,
            vwd_softcore_max
        ):
        """Generate all grids"""
        self._dists, self._elec_grid, self._vdw_grid = \
            self.comp_engine.gen_all_grids(
                self.coord_grid.coords, self.coords, self._charges, self._epsilons,
                self._vdw_rs, rad_dielec_const, elec_rep_max, elec_attr_max,
                probe_radius, vwd_softcore_max
            )

    def convert_to_boxed_grid(self, grid):
        """Convert a 1D grid to a 3D grid."""
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

    def save_dx(self, filename, grid_vals):
        """Save a grid to a .dx file."""
        boxed_grid = self.convert_to_boxed_grid(grid_vals)

        values_str = ''
        counter = 0
        for value in boxed_grid:
            counter += 1
            values_str += f'{value:e} ' 
            if counter % 6 == 0:
                values_str += '\n'

        dx_str = self._fill_dx(boxed_grid, values_str)
        with open(filename, 'w') as f:
            f.write(dx_str)

    def _fill_dx(self, grid, values_str):
        xd, yd, zd = self.coord_grid.points_per_dim
        min_x, min_y, min_z = self.coord_grid.min_coords
        spacing = self.spacing
        dx_template = (
            f'''#Generated dx file for fftgrid
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


class ProbeGridGenerator(EnerGridGenerator):
    """A potential energy grid generator for a small molecule probe."""
    # small molecule grids are computed on CPU for now
    comp_engine = GridCompEngine(backend='cpu')
    def __init__(self, grid_spacing, padding) -> None:
        super().__init__(grid_spacing, padding)
        self._grid_shape = "bounding_box"
        self.param_loader = ParameterLoader('cgenff')

    def load_entity(self, probe):
        """Load a probe and generate the grid coordinates and energy potentials
        associated with each grid point in space.
        
        Parameters
        ----------
        probe : :obj:`crimm.Data.probes._Probe` 
        The probe to be loaded.
        """
        super().load_entity(probe)
        self.coord_grid = self.bounding_box_grid
        self._collect_params()
        self._dists = self.comp_engine.cdist(
            self.coord_grid.coords, self.coords
        )
        self._elec_grid = None
        self._vdw_grids = None

    def _collect_params(self):
        epsilons = []
        vdw_rs = []
        charges = []
        for atom in self.entity.get_atoms():
            atom_type = atom.topo_definition.atom_type
            charges.append(atom.topo_definition.charge)
            epsilons.append(self.param_loader['nonbonded'][atom_type].epsilon)
            vdw_rs.append(self.param_loader['nonbonded'][atom_type].rmin_half)
        self._epsilons = np.asarray(epsilons)
        self._vdw_rs = np.asarray(vdw_rs)
        self._charges = np.asarray(charges)

    def _find_closest_vertices(self, abs_dists):
        dist2 = np.unique(np.sort(abs_dists))[:2]
        if 0 in dist2:
            return None
        return dist2

    def _get_cell_dist_vecs(self, abs_dist_vecs):
        # for possible scenario that an atom sits on a plane or on a grid point
        verts = []
        # for constructing 3D coords
        verts3 = []
        for dists in abs_dist_vecs.T:
            dist2 = self._find_closest_vertices(dists)
            if dist2 is not None:
                verts.append(dist2)
                verts3.append(dist2)
            else:
                verts3.append(np.zeros(2))
        return np.array(list(product(*verts))), np.array(list(product(*verts3)))

    def get_elec_grid(self):
        """Get the charges on the nearest grid points for each atom. The charges
        will be distributed to the grid points in a trilinear fashion."""
        if self._elec_grid is not None:
            return self._elec_grid
        grid_coords = self.coord_grid.coords
        charge_grid = np.zeros(grid_coords.shape[0])
        for i, coord in enumerate(self.coords):
            abs_dist_vecs = np.abs(grid_coords - coord)
            verts, verts3 = self._get_cell_dist_vecs(abs_dist_vecs)
            dist_ratios = verts/self.spacing
            cur_charges = self._charges[i]*np.prod(dist_ratios, 1)
            # Find the indices from the 3D coords
            # verts could be 2D or even 1D if the atom sits on a plane of a grid point
            indices = np.where(
                np.all(np.isin(abs_dist_vecs, verts3), axis = 1)
            )[0]
            charge_grid[indices] += cur_charges
        self._elec_grid = charge_grid
        return self._elec_grid

    def get_vdw_grid(self):
        raise NotImplementedError(
            'van der Waals grid is not implemented for small molecules yet.'
        )