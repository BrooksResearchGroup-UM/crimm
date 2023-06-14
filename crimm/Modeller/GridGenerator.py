import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull, Delaunay

class GridCoordGenerator:
    def __init__(self) -> None:
        self.entity = None
        self.coords = None
        self.resolution = None
        self.paddings = None
        self._convex_hull = None
        self._delaunay_on_convex_hull = None
        self._bounding_box_grid = None
        self._convex_hull_grid = None
        self._cubic_grid = None
        self._truncated_sphere_grid = None

    def load_entity(self, entity, grid_resolution, padding):
        self.entity = entity
        self.coords = self._extract_coords(self.entity)
        self.resolution = grid_resolution
        self.paddings = padding
        # remove all grid attributes if existing
        self._convex_hull = None
        self._delaunay_on_convex_hull = None
        self._bounding_box_grid = None
        self._convex_hull_grid = None
        self._cubic_grid = None
        self._truncated_sphere_grid = None

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
    def convex_hull_grid(self):
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
        self._cubic_grid = self._get_box_grid(
            self.coord_center, grid_half_widths, self.resolution
        )
        return self._bounding_box_grid

    def get_bounding_box_grid(self):
        """Return a grid of points (N, 3) that covers the bounding box of the 
        coordinates with paddings."""
        grid_half_widths = np.ceil(self.box_dim/2)+self.paddings
        self._bounding_box_grid = self._get_box_grid(
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
        return grid_pos

    def get_truncated_sphere_grid(self):
        """Return a grid of points (N, 3) that covers the truncated sphere of
        the coordinates with paddings."""
        bounding_box_grid = self.bounding_box_grid
        radius = np.ceil(self.box_dim[0]/2)+self.paddings
        # Euclidean distance normalized by semi-axes
        distances = np.linalg.norm(
            (bounding_box_grid-self.coord_center) / radius, axis=1
        )
        self._truncated_sphere_grid = bounding_box_grid[distances <= 1]
        # points_shell = bounding_box_grid[np.abs(distances - 1) <= 1e-3]
        return self._truncated_sphere_grid

    def get_enlarged_convex_hull_grid(self):
        """Return a grid of points (N, 3) that covers the an enlarged convex hull.
        The enlarged convex hull is defined as the convex hull of the coordinates
        with paddings."""

        hull = self._enlarged_convex_hull()
        bounding_box_grid = self.bounding_box_grid
        # Find the Delaunay triangulation of the convex hull
        if self._delaunay_on_convex_hull is None:
            self._delaunay_on_convex_hull = Delaunay(hull)
        hull_grid_ids = np.argwhere(
            self._delaunay_on_convex_hull.find_simplex(bounding_box_grid) >= 0
        ).reshape(-1)
        self._convex_hull_grid = bounding_box_grid[hull_grid_ids]
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
        enlarged_hull = self.coords[hull.vertices] + displacement
        return enlarged_hull

    def find_hull_simplex_normals(self, hull):
        simplices_coords = hull.points[hull.simplices]
        normals = np.cross(
            simplices_coords[:, 0]-simplices_coords[:, 2],
            simplices_coords[:, 1]-simplices_coords[:, 2],
            axis=1
        )
        # normalize the vector normals
        normals = (normals.T/norm(normals, axis=1)).T
        return normals

    def find_vec_coords_to_simplices(self, vert_coords, coords):
        n_verts = vert_coords.shape[0]
        n_coords = coords.shape[0]
        verts_expanded = np.repeat(
            vert_coords, n_coords, axis=1
        ).reshape(*vert_coords.shape, n_coords)

        coords_expanded = np.repeat(
            coords, n_verts, axis=1
        ).reshape(*coords.shape, n_verts).T

        coords_to_simplices = coords_expanded - verts_expanded
        coords_to_simplices = np.einsum('ijk->ikj', coords_to_simplices)
        return coords_to_simplices

class _Grid:
    def __init__(self) -> None:
        self.grid_type = None
        self.coords = None
        self.surf_coords = None
        self.surf_normals = None
        self.surf_ids = None

