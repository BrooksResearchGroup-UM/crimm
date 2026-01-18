"""Signed Distance Functions (SDF) for CHARMM crystal geometries.

Each SDF returns negative values inside the shape, zero on the boundary,
and positive values outside. The magnitude represents distance to the surface.

These functions are used to determine which water molecules fall within
a given crystal geometry during solvation.

Based on CHARMM source code:
- ~/software/charmm/source/image/pbound.F90 (minimum image conventions)
- ~/software/charmm/source/image/crystal.F90 (crystal definitions)

Reference:
M. P. Allen and D. J. Tildesley, Computer Simulation of Liquids, Ch. 1.
"""

import math
import numpy as np
from typing import Tuple

# Constants
SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT_HALF = math.sqrt(0.5)
SQRT_4_3 = math.sqrt(4.0 / 3.0)


def sdf_cube(point: np.ndarray, box_dim: float) -> float:
    """SDF for cubic box centered at origin.

    Parameters
    ----------
    point : np.ndarray
        3D coordinate (x, y, z)
    box_dim : float
        Side length of the cube

    Returns
    -------
    float
        Signed distance (negative inside, positive outside)
    """
    half = box_dim / 2.0
    x, y, z = np.abs(point)
    return max(x - half, y - half, z - half)


def sdf_truncated_octahedron(point: np.ndarray, box_dim: float) -> float:
    """SDF for truncated octahedron (CHARMM OCTA).

    The truncated octahedron has 14 faces: 6 square faces (along axes)
    and 8 hexagonal faces (at corners, cutting the cube corners).

    Volume = (4*sqrt(3)/9) * a^3 ≈ 0.77 * a^3

    Parameters
    ----------
    point : np.ndarray
        3D coordinate (x, y, z)
    box_dim : float
        The 'a' parameter (a=b=c for this crystal type)

    Returns
    -------
    float
        Signed distance (negative inside, positive outside)
    """
    x, y, z = point

    # Square face distance (along each axis)
    # d = box_dim / sqrt(3) is the half-width along each axis
    d = box_dim / SQRT3
    dist_square = max(abs(x) - d, abs(y) - d, abs(z) - d)

    # Hexagonal face distance (cutting the corners)
    # These are planes at the 8 corners of a cube
    dist_hex = max(
        (abs(x + y + z) - box_dim) / SQRT3,
        (abs(x + y - z) - box_dim) / SQRT3,
        (abs(x - y + z) - box_dim) / SQRT3,
        (abs(x - y - z) - box_dim) / SQRT3
    )

    return max(dist_square, dist_hex)


def sdf_rhombic_dodecahedron(point: np.ndarray, box_dim: float) -> float:
    """SDF for rhombic dodecahedron (CHARMM RHDO).

    The rhombic dodecahedron has 12 rhombic faces. It can be defined as
    the intersection of 6 slabs oriented along face diagonals of a cube.

    Volume = sqrt(0.5) * a^3 ≈ 0.707 * a^3
    Bounding cube length = a * sqrt(2)

    Based on CHARMM pbound.F90 PBMove routine (qRDBoun).

    Parameters
    ----------
    point : np.ndarray
        3D coordinate (x, y, z)
    box_dim : float
        The 'a' parameter (a=b=c for this crystal type)

    Returns
    -------
    float
        Signed distance (negative inside, positive outside)
    """
    x, y, z = point

    # The rhombic dodecahedron can be defined by 6 face-diagonal planes
    # Each pair of faces is defined by |±x ± y|, |±y ± z|, |±z ± x|
    # The half-width in each direction is a/2
    half = box_dim / 2.0

    # Six pairs of parallel rhombic faces
    dist = max(
        abs(x + y) / SQRT2 - half,
        abs(x - y) / SQRT2 - half,
        abs(y + z) / SQRT2 - half,
        abs(y - z) / SQRT2 - half,
        abs(z + x) / SQRT2 - half,
        abs(z - x) / SQRT2 - half,
    )

    return dist


def sdf_orthorhombic(point: np.ndarray, dims: Tuple[float, float, float]) -> float:
    """SDF for orthorhombic box (rectangular prism) with different dimensions.

    alpha = beta = gamma = 90 degrees, but a != b != c

    Parameters
    ----------
    point : np.ndarray
        3D coordinate (x, y, z)
    dims : tuple of float
        Box dimensions (a, b, c) in Angstroms

    Returns
    -------
    float
        Signed distance (negative inside, positive outside)
    """
    x, y, z = point
    a, b, c = dims
    return max(abs(x) - a / 2.0, abs(y) - b / 2.0, abs(z) - c / 2.0)


def sdf_tetragonal(point: np.ndarray, a: float, c: float) -> float:
    """SDF for tetragonal box (a = b != c, all angles 90 degrees).

    Parameters
    ----------
    point : np.ndarray
        3D coordinate (x, y, z)
    a : float
        The a=b dimension
    c : float
        The c dimension

    Returns
    -------
    float
        Signed distance (negative inside, positive outside)
    """
    return sdf_orthorhombic(point, (a, a, c))


def sdf_hexagonal(point: np.ndarray, a: float, c: float) -> float:
    """SDF for hexagonal prism (a=b, alpha=beta=90°, gamma=120°).

    The cross-section is a regular hexagon with circumradius related to 'a'.
    Height is 'c'.

    Based on CHARMM pbound.F90 qRHBoun (2D rhomboidal for hexagonal base).

    Parameters
    ----------
    point : np.ndarray
        3D coordinate (x, y, z)
    a : float
        The a=b dimension (hexagon circumradius)
    c : float
        The c dimension (height)

    Returns
    -------
    float
        Signed distance (negative inside, positive outside)
    """
    x, y, z = point

    # Height constraint
    dist_z = abs(z) - c / 2.0

    # Hexagonal cross-section
    # A regular hexagon can be defined by 3 pairs of parallel lines
    # The inradius (distance from center to edge) is a * sqrt(3) / 2
    # We use the circumradius 'a' as the defining parameter
    inradius = a * SQRT3 / 2.0

    # Three pairs of parallel edges
    dist_hex = max(
        abs(x) - inradius,
        abs(x * 0.5 + y * SQRT3 / 2.0) - inradius,
        abs(x * 0.5 - y * SQRT3 / 2.0) - inradius,
    )

    return max(dist_z, dist_hex)


# Vectorized versions for batch processing

def sdf_cube_batch(points: np.ndarray, box_dim: float) -> np.ndarray:
    """Vectorized SDF for cubic box.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) with 3D coordinates
    box_dim : float
        Side length of the cube

    Returns
    -------
    np.ndarray
        Array of signed distances for each point
    """
    half = box_dim / 2.0
    abs_points = np.abs(points)
    return np.maximum.reduce([
        abs_points[:, 0] - half,
        abs_points[:, 1] - half,
        abs_points[:, 2] - half
    ])


def sdf_truncated_octahedron_batch(points: np.ndarray, box_dim: float) -> np.ndarray:
    """Vectorized SDF for truncated octahedron.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) with 3D coordinates
    box_dim : float
        The 'a' parameter

    Returns
    -------
    np.ndarray
        Array of signed distances for each point
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    d = box_dim / SQRT3
    dist_square = np.maximum.reduce([np.abs(x) - d, np.abs(y) - d, np.abs(z) - d])

    dist_hex = np.maximum.reduce([
        (np.abs(x + y + z) - box_dim) / SQRT3,
        (np.abs(x + y - z) - box_dim) / SQRT3,
        (np.abs(x - y + z) - box_dim) / SQRT3,
        (np.abs(x - y - z) - box_dim) / SQRT3
    ])

    return np.maximum(dist_square, dist_hex)


def sdf_rhombic_dodecahedron_batch(points: np.ndarray, box_dim: float) -> np.ndarray:
    """Vectorized SDF for rhombic dodecahedron.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) with 3D coordinates
    box_dim : float
        The 'a' parameter

    Returns
    -------
    np.ndarray
        Array of signed distances for each point
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    half = box_dim / 2.0

    return np.maximum.reduce([
        np.abs(x + y) / SQRT2 - half,
        np.abs(x - y) / SQRT2 - half,
        np.abs(y + z) / SQRT2 - half,
        np.abs(y - z) / SQRT2 - half,
        np.abs(z + x) / SQRT2 - half,
        np.abs(z - x) / SQRT2 - half,
    ])


def sdf_orthorhombic_batch(points: np.ndarray, dims: Tuple[float, float, float]) -> np.ndarray:
    """Vectorized SDF for orthorhombic box.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) with 3D coordinates
    dims : tuple of float
        Box dimensions (a, b, c)

    Returns
    -------
    np.ndarray
        Array of signed distances for each point
    """
    a, b, c = dims
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.maximum.reduce([
        np.abs(x) - a / 2.0,
        np.abs(y) - b / 2.0,
        np.abs(z) - c / 2.0
    ])


def sdf_hexagonal_batch(points: np.ndarray, a: float, c: float) -> np.ndarray:
    """Vectorized SDF for hexagonal prism.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) with 3D coordinates
    a : float
        The a=b dimension
    c : float
        The c dimension

    Returns
    -------
    np.ndarray
        Array of signed distances for each point
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    dist_z = np.abs(z) - c / 2.0

    inradius = a * SQRT3 / 2.0
    dist_hex = np.maximum.reduce([
        np.abs(x) - inradius,
        np.abs(x * 0.5 + y * SQRT3 / 2.0) - inradius,
        np.abs(x * 0.5 - y * SQRT3 / 2.0) - inradius,
    ])

    return np.maximum(dist_z, dist_hex)


# Registry mapping box types to their SDF functions
SDF_FUNCTIONS = {
    'cube': sdf_cube,
    'octa': sdf_truncated_octahedron,
    'rhdo': sdf_rhombic_dodecahedron,
    'ortho': sdf_orthorhombic,
    'tetra': sdf_tetragonal,
    'hexa': sdf_hexagonal,
}

SDF_FUNCTIONS_BATCH = {
    'cube': sdf_cube_batch,
    'octa': sdf_truncated_octahedron_batch,
    'rhdo': sdf_rhombic_dodecahedron_batch,
    'ortho': sdf_orthorhombic_batch,
    'hexa': sdf_hexagonal_batch,
}


def get_bounding_cube_size(box_type: str, box_dim: float) -> float:
    """Get the bounding cube size for a given crystal type.

    This is used to determine the initial water grid size before
    filtering by the crystal SDF.

    Parameters
    ----------
    box_type : str
        Crystal type ('cube', 'octa', 'rhdo', etc.)
    box_dim : float
        The primary box dimension (a parameter)

    Returns
    -------
    float
        Side length of the bounding cube
    """
    if box_type == 'cube':
        return box_dim
    elif box_type == 'octa':
        # Truncated octahedron bounding cube: a * sqrt(4/3)
        return box_dim * SQRT_4_3
    elif box_type == 'rhdo':
        # Rhombic dodecahedron bounding cube: a * sqrt(2)
        return box_dim * SQRT2
    else:
        # For other types, use the box_dim directly
        return box_dim
