import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull, Delaunay
from crimm.Visualization import View
from crimm.Modeller import ParameterLoader
from crimm.Data.constants import cc_elec_charmm as cc_elec
from crimm.GridGen._grid_gen_wrappers import (
    cdist_wrapper, gen_elec_grid_wrapper, gen_vdw_grid_wrapper
)
class GridCoordGenerator:
    def __init__(self) -> None:
        self.entity = None
        self.coords = None
        self.resolution = None
        self.paddings = None
        self._convex_hull = None # ConvexHull (scipy.Qhull) object
        self._enlarged_hull_vertices = None
        # grid ids of the enlarged convex hull within the bounding box grid
        self._enlarged_hull_grid_ids = None
        self._delaunay_on_convex_hull = None
        self._bounding_box_grid = None
        self._convex_hull_grid = None # enlarged convex hull grid
        self._hull_surf_coords = None # coordinates of the enlarged convex hull surface
        self._cubic_grid = None
        self._truncated_sphere_grid = None
        self._truc_sphere_shell = None
        self._simplices_normals = None
        self._verts_i = None
        self._n_points_per_dim = None # maximum number of the grid points in each dimension

    def load_entity(self, entity, grid_resolution, padding):
        """Load entity and set grid resolution and paddings."""
        self.entity = entity
        self.coords = self._extract_coords(self.entity)
        self.resolution = grid_resolution
        self.paddings = padding
        # remove all grid attributes if existing
        self._convex_hull = None
        self._enlarged_hull_vertices = None
        self._enlarged_hull_grid_ids = None
        self._delaunay_on_convex_hull = None
        self._bounding_box_grid = None
        self._convex_hull_grid = None
        self._hull_surf_coords = None
        self._cubic_grid = None
        self._truncated_sphere_grid = None
        self._truc_sphere_shell = None
        self._simplices_normals = None
        self._verts_i = None
        self._n_points_per_dim = None

    def _extract_coords(self, entity) -> np.array:
        coords = []
        for atom in entity.get_atoms():
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
    def box_dim(self):
        """Return the dimensions of the bounding box of the coordinates (N, 3).
        The three sides of the box are parallel to the x, y, and z axes.
        """
        return self.coords.ptp(0)

    @property
    def convex_hull(self):
        """Return the convex hull of the coordinates (N, 3)."""
        if self._convex_hull is None:
            self._convex_hull = ConvexHull(self.coords)
        return self._convex_hull

    @property
    def truncated_sphere_grid(self):
        """Return the truncated sphere of the coordinates (N, 3) with paddings."""
        if self._truncated_sphere_grid is None:
            return self.get_truncated_sphere_grid()
        return self._truncated_sphere_grid

    @property
    def truncated_sphere_shell(self):
        """Return the coordinate (N, 3) of the shell truncated sphere with paddings."""
        if self._truc_sphere_shell is None:
            self.get_truncated_sphere_grid()
        return self._truc_sphere_shell

    @property
    def enlarged_convex_hull_grid(self):
        """Return a grid of points (N, 3) that covers the convex hull of the coordinates."""
        if self._convex_hull_grid is None:
            return self.get_enlarged_convex_hull_grid()
        return self._convex_hull_grid

    @property
    def bounding_box_grid(self):
        """Return a grid of points (N, 3) that covers the bounding box of the coordinates."""
        if self._bounding_box_grid is None:
            return self.get_bounding_box_grid()
        return self._bounding_box_grid

    @property
    def cubic_grid(self):
        """Return a grid of points (N, 3) that covers the bounding cube of the coordinates."""
        if self._cubic_grid is None:
            return self.get_bounding_cube_grid()
        return self._cubic_grid

    def get_bounding_cube_grid(self):
        """Return a grid of points (N, 3) that covers the bounding cube of the 
        coordinates with paddings."""
        grid_half_widths = (np.ceil(self.box_dim[0]/2)+self.paddings)*np.ones(3)
        self._n_points_per_dim, self._cubic_grid = self._get_box_grid(
            self.coord_center, grid_half_widths, self.resolution
        )
        return self._bounding_box_grid

    def get_bounding_box_grid(self):
        """Return a grid of points (N, 3) that covers the bounding box of the 
        coordinates with paddings."""
        grid_half_widths = np.ceil(self.box_dim/2)+self.paddings
        self._n_points_per_dim, self._bounding_box_grid = self._get_box_grid(
            self.coord_center, grid_half_widths, self.resolution
        )
        return self._bounding_box_grid

    @staticmethod
    def _get_box_grid(center, grid_half_widths, resolution):
        """Return a grid of points (N, 3) defined by a center (x, y, z) and the
        grid box's half widths (x_len/2, y_len/2, z_len/2)."""
        dims = []
        for mid_point, half_width in zip(center, grid_half_widths):
            dims.append(
                np.arange(
                    mid_point-half_width,
                    mid_point+half_width+resolution,
                    resolution
                )
            )
        grid_pos = np.array(np.meshgrid(*dims, indexing='ij')).reshape(3,-1).T
        x_pos, y_pos, z_pos = dims
        dim_sizes = np.array([x_pos.size, y_pos.size, z_pos.size])
        return dim_sizes, grid_pos

    def get_truncated_sphere_grid(self):
        """Return a grid of points (N, 3) that covers the truncated sphere of
        the coordinates with paddings."""
        tolerance = self.resolution*1e-2
        bounding_box_grid = self.bounding_box_grid
        radius = np.ceil(max(self.box_dim)/2)+self.paddings
        # Euclidean distance normalized by semi-axes
        distances = np.linalg.norm(
            (bounding_box_grid-self.coord_center) / radius, axis=1
        )
        self._truncated_sphere_grid = bounding_box_grid[distances <= 1]
        self._truc_sphere_shell = bounding_box_grid[
            np.abs(distances - 1) <= tolerance
        ]
        return self._truncated_sphere_grid

    def get_enlarged_convex_hull_grid(self):
        """Return a grid of points (N, 3) that covers the an enlarged convex hull.
        The enlarged convex hull is defined as the convex hull of the coordinates
        with paddings."""
        if self._enlarged_hull_vertices is None:
            self._enlarged_convex_hull()
        hull = self._enlarged_hull_vertices
        # Find the Delaunay triangulation of the convex hull
        if self._delaunay_on_convex_hull is None:
            self._delaunay_on_convex_hull = Delaunay(hull)
        hull_grid_ids = np.argwhere(
            self._delaunay_on_convex_hull.find_simplex(
                self.bounding_box_grid
            ) >= 0
        ).reshape(-1)
        self._convex_hull_grid = self.bounding_box_grid[hull_grid_ids]
        self._enlarged_hull_grid_ids = hull_grid_ids
        return self._convex_hull_grid

    def _enlarged_convex_hull(self):
        """Return the vertices of the enlarged convex hull of the coordinates"""
        # Compute the convex hull if not already computed
        hull = self.convex_hull
        # Find the centroid of the convex hull
        centroid = np.mean(self.coords[hull.vertices], axis=0)
        # Compute the vectors from the centroid to each vertex
        vectors = self.coords[hull.vertices] - centroid
        # Normalize the vectors to unit length
        norms = np.linalg.norm(vectors, axis=1)
        normalized_vectors = vectors / norms[:, np.newaxis]
        # Compute the displacement vector for each vertex
        displacement = normalized_vectors * self.paddings
        # Enlarge the convex hull by adding the displacement vector to each vertex
        self._enlarged_hull_vertices = self.coords[hull.vertices] + displacement
        return self._enlarged_hull_vertices

    def _find_hull_simplex_normals(self):
        hull = self.convex_hull
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

        tolerance = self.resolution*1e-2
        hull_grid = self.enlarged_convex_hull_grid
        dists = self.find_coords_to_simplices_dists(hull_grid)
        surf_ids = np.argwhere(
            np.any(np.abs(dists)<tolerance, axis=0)
        ).reshape(-1)
        self._hull_surf_coords = hull_grid[surf_ids]
        return self._hull_surf_coords

    def show_hull_surface(self, show_licorice=False, show_enlarged_hull=False):
        """Show the surface of the convex hull."""
        hull = self.convex_hull
        if show_enlarged_hull:
            if self._enlarged_hull_vertices is None:
                self._enlarged_convex_hull()
            idx_dict = {x: idx for idx, x in enumerate(hull.vertices)}
            enlarged_simplex_ids = np.vectorize(idx_dict.get)(hull.simplices)
            flattened_array = (
                self._enlarged_hull_vertices[enlarged_simplex_ids].reshape(-1)
            )
        else:
            flattened_array = hull.points[hull.simplices].reshape(-1)
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
    
class EnerGridGenerator(GridCoordGenerator):
    grid_shape_dict = {
        'cubic': 'cubic_grid',
        'bounding_box': 'bounding_box_grid',
        'truncated_sphere': 'truncated_sphere_grid',
        'convex_hull' : 'enlarged_convex_hull_grid'
    }

    def __init__(self) -> None:
        super().__init__()
        self._grid_coords = None
        self._elec_grid = None
        self._vdw_grid = None
        self._dists = None
        self._param_loader = None
        self._charges = None
        self._epsilons = None
        self._vdw_rs = None
        self._min_coords = None
        self._grid_shape = None

    def load_entity(
            self, entity, grid_resolution, padding, 
            grid_shape = 'convex_hull'
        ):
        """Load an entity and generate the grid coordinates and energy potentials
        associated with each grid point in space.
        
        Parameters
        ----------
        entity : :obj:`crimm.StructEntity.Chain` 
        The entity to be loaded.
        grid_resolution : float
        The resolution of the grid in Angstroms.
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
            self._param_loader = ParameterLoader('protein')
        elif entity.chain_type == "Polyribonucleotide":
            self._param_loader = ParameterLoader('nucleic')
        else:
            raise ValueError(
                f'entity must be a protein or an RNA, got {self.entity.chain_type}'
            )

        super().load_entity(entity, grid_resolution, padding)
        self._grid_shape = grid_shape
        self._grid_coords = getattr(self, self.grid_shape_dict[grid_shape])
        if grid_shape in ('convex_hull', 'truncate_sphere'):
            self._min_coords = np.min(self.bounding_box_grid, axis=0)
        else:
            self._min_coords = np.min(self._grid_coords, axis=0)
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
            nb_param = self._param_loader['nonbonded'][atom_type]
            vdw_rs. append(nb_param.rmin_half)
            epsilons.append(nb_param.epsilon)

        self._charges = np.array(charges)
        self._vdw_rs = np.array(vdw_rs)
        self._epsilons = np.array(epsilons)

    def get_pairwise_dists(self):
        if self._dists is None:
            self._dists = cdist_wrapper(self._grid_coords, self.coords)
        return self._dists
 
    def get_vdw_grid(self, probe_radius, vwd_softcore_max):
        """Get the van der Waals energy grid."""
        pairwise_dists = self.get_pairwise_dists()
        if self._vdw_grid is None:
            self._vdw_grid = gen_vdw_grid_wrapper(
                pairwise_dists, self._epsilons, self._vdw_rs, 
                probe_radius, vwd_softcore_max
            )
        return self._vdw_grid

    def get_elec_grid(self, rad_dielec_const, elec_rep_max, elec_attr_max):
        """Get the electrostatic energy grid."""
        pairwise_dists = self.get_pairwise_dists()
        if self._elec_grid is None:
            self._elec_grid = gen_elec_grid_wrapper(
                pairwise_dists, self._charges, cc_elec, rad_dielec_const,
                elec_rep_max, elec_attr_max
            )
        return self._elec_grid

    def _place_grid_back_in_box(self, grid):
        boxed_grid = np.zeros(self.bounding_box_grid.shape[0])
        if self._enlarged_hull_grid_ids is None:
            self.get_enlarged_convex_hull_grid()
        filled_ids = self._enlarged_hull_grid_ids
        boxed_grid[filled_ids] = grid
        return boxed_grid
    
    def save_dx(self, filename, grid):
        """Save a grid to a .dx file."""
        if self._grid_shape in ('convex_hull', 'truncated_sphere'):
            boxed_grid = self._place_grid_back_in_box(grid)
        else:
            boxed_grid = grid
        
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
        xd, yd, zd = self._n_points_per_dim
        min_x, min_y, min_z = self._min_coords
        spacing = self.resolution
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