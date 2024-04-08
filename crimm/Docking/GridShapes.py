import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull, Delaunay

def is_divisible_by_2357(x):
    """Check if the input integer is divisible by 2, 3, 5, or 7."""
    prime_factors = np.array([2, 3, 5, 7])
    modulus = np.mod(x, prime_factors)
    mask = np.logical_not(modulus)
    if not mask.any():
        return False
    other_factors = x//prime_factors[mask]
    if 1 in other_factors:
        return True
    for factor in other_factors:
        return is_divisible_by_2357(factor)

def find_optimized_dim(x):
    """Return the smallest number x' that is divisible by 2, 3, 5, or 7 
    where x' >= x (the input number)."""
    while not is_divisible_by_2357(x):
        x+=1
    return x

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

    def find_optimal_dims(self, dims):
        """Return the optimal dimensions for fast fourier transform. The returned
        dimensions are the smallest power of (2, 3, 5, 7) that is greater than 
        the input dims."""
        for i, dim in enumerate(dims):
            dims[i] = find_optimized_dim(dim)
        return dims

    def _get_box_grid(
            self, center, grid_half_widths, spacing, optimize_for_fft=False
        ):
        """Return a grid of points (N, 3) defined by a center (x, y, z) and the
        grid box's half widths (x_len/2, y_len/2, z_len/2)."""
        dims = []
        if optimize_for_fft:
            widths = grid_half_widths*2
            n_grids = np.ceil(widths/spacing).astype(int)
            n_grids = self.find_optimal_dims(n_grids)-1
            grid_half_widths = n_grids*spacing/2
        for mid_point, half_width in zip(center, grid_half_widths):
            dims.append(
                np.arange(
                    mid_point-half_width,
                    mid_point+half_width+spacing,
                    spacing
                )
            )
        x_pos, y_pos, z_pos = dims
        dim_sizes = np.array([x_pos.size, y_pos.size, z_pos.size])
        if optimize_for_fft:
            assert np.alltrue(dim_sizes == n_grids+1)
        grid_pos = np.ascontiguousarray(np.array(
            np.meshgrid(*dims, indexing='ij')
        ).reshape(3,-1).T, dtype=np.float32)
        
        return dim_sizes, grid_pos

class CubeGrid(_Grid):
    """A grid of points (N, 3) that covers the bounding cube of the coordinates."""
    def __init__(self, dims, coord_center, spacing, padding, optimize_for_fft) -> None:
        super().__init__(coord_center, spacing, padding)
        grid_half_widths = (np.ceil(np.max(dims)/2)+padding)*np.ones(3)
        self._points_per_dim, self._coords = self._get_box_grid(
            coord_center, grid_half_widths, spacing, optimize_for_fft
        )
        self.min_coords = self._coords.min(0)

class BoundingBoxGrid(_Grid):
    """A grid of points (N, 3) that covers the bounding box of the coordinates."""
    def __init__(self, dims, coord_center, spacing, padding, optimize_for_fft) -> None:
        super().__init__(coord_center, spacing, padding)
        grid_half_widths = np.ceil(dims/2)+self._padding
        self._points_per_dim, self._coords = self._get_box_grid(
            coord_center, grid_half_widths, spacing, optimize_for_fft
        )
        self.min_coords = self._coords.min(0)

class TruncatedSphereGrid(BoundingBoxGrid):
    """A grid of points (N, 3) that covers the truncated sphere of the coordinates."""
    def __init__(self, dims, coord_center, spacing, padding, optimize_for_fft) -> None:
        super().__init__(dims, coord_center, spacing, padding, optimize_for_fft)
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
    def __init__(self, entity_coords, dims, coord_center, spacing, padding, optimize_for_fft) -> None:
        super().__init__(dims, coord_center, spacing, padding, optimize_for_fft)
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