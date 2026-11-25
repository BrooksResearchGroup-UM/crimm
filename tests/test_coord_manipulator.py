"""CoordManipulator: orientation must be a rigid, proper transformation."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from crimm.Modeller.CoordManipulator import CoordManipulator

ORIENT_METHODS = [
    "orient_coords",
    "orient_coords_octa",
    "orient_coords_ortho",
    "orient_coords_hexa",
]
PCA_METHODS = ORIENT_METHODS[1:]
N_SEEDS = 20


class FakeAtom:
    def __init__(self, coord):
        self.coord = coord


class FakeEntity:
    """The minimum CoordManipulator needs: get_atoms() and a parent."""

    parent = None

    def __init__(self, coords):
        self.atoms = [FakeAtom(coord) for coord in coords]

    def get_atoms(self, include_alt=False):
        return self.atoms


def random_cloud(seed, n=60):
    """Anisotropic point cloud, so the principal axes are well defined."""
    return np.random.default_rng(seed).normal(size=(n, 3)) * [6.0, 3.0, 1.5]


def orient(method, coords):
    entity = FakeEntity(coords.copy())
    manipulator = CoordManipulator()
    manipulator.load_entity(entity)
    getattr(manipulator, method)()
    return manipulator, np.array([atom.coord for atom in entity.atoms])


def handedness(coords):
    """Sign of the volume spanned by the first four points; flips under reflection."""
    return np.sign(np.linalg.det(coords[1:4] - coords[0]))


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_orientation_preserves_distances(method):
    coords = random_cloud(0)
    _, oriented = orient(method, coords)
    assert pdist(oriented) == pytest.approx(pdist(coords), abs=1e-3)


@pytest.mark.parametrize("method", PCA_METHODS)
def test_pca_orientation_centres_on_centroid(method):
    _, oriented = orient(method, random_cloud(1))
    assert oriented.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-6)


def test_default_orientation_centres_bounding_box():
    manipulator, oriented = orient("orient_coords", random_cloud(2))
    centre = (oriented.max(axis=0) + oriented.min(axis=0)) / 2
    assert centre == pytest.approx([0, 0, 0], abs=1e-3)
    assert manipulator.coord_center == pytest.approx([0, 0, 0], abs=1e-3)


def test_default_orientation_puts_longest_axis_on_x():
    _, oriented = orient("orient_coords", random_cloud(3))
    extent = np.ptp(oriented, axis=0)
    assert extent[0] == extent.max()


@pytest.mark.parametrize("method", ["orient_coords_ortho", "orient_coords_hexa"])
def test_sorted_pca_orientation_orders_extents(method):
    _, oriented = orient(method, random_cloud(4))
    spread = oriented.std(axis=0)
    assert spread[0] >= spread[1] >= spread[2]


@pytest.mark.parametrize(
    "method",
    [
        "orient_coords",
        pytest.param(
            "orient_coords_octa",
            marks=pytest.mark.xfail(
                strict=True,
                reason="#43: no determinant check on the PCA axes: about half of all inputs "
                "are reflected, which inverts chirality. This is the default "
                "orientation for cube, octa and rhdo solvation boxes",
            ),
        ),
        "orient_coords_ortho",
        "orient_coords_hexa",
    ],
)
def test_orientation_preserves_handedness(method):
    flipped = [
        seed
        for seed in range(N_SEEDS)
        if handedness(orient(method, random_cloud(seed))[1])
        != handedness(random_cloud(seed))
    ]
    assert flipped == []


@pytest.mark.parametrize(
    "method",
    [
        "orient_coords",
        *[
            pytest.param(
                name,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="#44: PCA methods store op_mat in a row-vector layout that "
                    "apply_coords reads as column-vector",
                ),
            )
            for name in PCA_METHODS
        ],
    ],
)
def test_stored_matrix_reproduces_the_orientation(method):
    coords = random_cloud(5)
    manipulator, oriented = orient(method, coords)
    assert manipulator.apply_coords(coords) == pytest.approx(oriented, abs=1e-3)


def test_orient_without_entity_raises():
    with pytest.raises(ValueError):
        CoordManipulator().orient_coords()


@pytest.mark.xfail(
    np.lib.NumpyVersion(np.__version__) >= "2.0.0",
    strict=True,
    reason="#51: box_dim calls ndarray.ptp, which NumPy 2 removed",
)
def test_box_dim():
    coords = random_cloud(6)
    manipulator, oriented = orient("orient_coords", coords)
    assert manipulator.box_dim == pytest.approx(np.ptp(oriented, axis=0), abs=1e-3)


def test_orientation_of_a_real_structure(load_structure):
    model = load_structure("1ubq").models[0]
    before = np.array([atom.coord for atom in model.get_atoms()])
    manipulator = CoordManipulator()
    manipulator.load_entity(model)
    manipulator.orient_coords()
    after = np.array([atom.coord for atom in model.get_atoms()])
    assert pdist(after[:150]) == pytest.approx(pdist(before[:150]), abs=1e-2)
    assert handedness(after) == handedness(before)


def test_find_max_dim_is_the_largest_interatomic_distance():
    coords = random_cloud(7)
    manipulator = CoordManipulator()
    manipulator.load_entity(FakeEntity(coords))
    max_dim = manipulator.find_max_dim()
    assert isinstance(max_dim, float)
    assert max_dim == pytest.approx(pdist(coords).max(), rel=1e-6)


def test_find_max_dim_is_unchanged_by_orientation():
    coords = random_cloud(8)
    _, oriented = orient("orient_coords", coords)
    reloaded = CoordManipulator()
    reloaded.load_entity(FakeEntity(oriented))
    assert reloaded.find_max_dim() == pytest.approx(pdist(coords).max(), abs=1e-3)


def test_find_max_dim_without_entity_raises():
    with pytest.raises(ValueError):
        CoordManipulator().find_max_dim()


def brute_force_farthest_pair(coords):
    """The pair `load_entity` used to find, from the full distance matrix."""
    dist_matrix = squareform(pdist(coords))
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return (int(i), int(j)), float(dist_matrix[i, j])


def assert_matches_brute_force(coords):
    manipulator = CoordManipulator()
    manipulator.load_entity(FakeEntity(coords))
    pair, max_dim = brute_force_farthest_pair(coords)
    assert (manipulator.end_i, manipulator.end_j) == pair
    assert manipulator.find_max_dim() == max_dim
    assert manipulator._dist_matrix is None, "the N x N matrix must not be built"


def lattice(n):
    """n**3 grid points: many coplanar hull points and many tied farthest pairs."""
    axis = np.arange(n, dtype=float)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [5, 30, 500, 3000])
def test_farthest_pair_matches_brute_force_on_random_sets(seed, n):
    assert_matches_brute_force(random_cloud(seed, n))


@pytest.mark.parametrize("seed", range(5))
def test_farthest_pair_matches_brute_force_in_single_precision(seed):
    # atom coordinates are float32
    assert_matches_brute_force(random_cloud(seed, 2000).astype(np.float32))


@pytest.mark.parametrize("pdb_id", ["1ubq", "1crn", "3ptb", "1bna"])
def test_farthest_pair_matches_brute_force_on_structures(load_structure, pdb_id):
    model = load_structure(pdb_id).models[0]
    coords = np.array([atom.coord for atom in model.get_atoms()])
    assert_matches_brute_force(coords)


rng = np.random.default_rng(52)
DEGENERATE = {
    "one point": np.ones((1, 3)),
    "two points": np.array([[0.0, 0, 0], [1, 2, 3]]),
    "three points": rng.normal(size=(3, 3)),
    "four points": rng.normal(size=(4, 3)),
    "identical points": np.tile([1.5, -2.0, 3.0], (10, 1)),
    "collinear": np.outer(rng.normal(size=50), [1.0, 2.0, -0.5]),
    "coplanar": np.column_stack([rng.normal(size=(200, 2)), np.zeros(200)]),
    "tilted plane": rng.normal(size=(200, 2)) @ [[1.0, 0.3, -2.0], [0.5, 1.0, 1.0]],
    "cube corners": lattice(2),
    "lattice": lattice(6),
    "duplicated ends": np.vstack([random_cloud(9, 100)] * 3),
    "regular polygon": np.column_stack(
        [np.cos(np.arange(12) * np.pi / 6), np.sin(np.arange(12) * np.pi / 6), [0] * 12]
    ),
}


@pytest.mark.parametrize("name", DEGENERATE)
def test_farthest_pair_matches_brute_force_on_degenerate_sets(name):
    assert_matches_brute_force(DEGENERATE[name])


def test_farthest_pair_is_not_computed_by_load_entity(monkeypatch):
    def fail(self):
        raise AssertionError("farthest pair computed although not needed")

    monkeypatch.setattr(CoordManipulator, "_find_farthest_atom_indices", fail)
    manipulator = CoordManipulator()
    manipulator.load_entity(FakeEntity(random_cloud(10)))
    manipulator.orient_coords_ortho()


def test_farthest_pair_is_that_of_the_loaded_coordinates():
    coords = random_cloud(11)
    pair, _ = brute_force_farthest_pair(coords)
    manipulator = CoordManipulator()
    manipulator.load_entity(FakeEntity(coords.copy()))
    manipulator.orient_coords_octa()
    assert (manipulator.end_i, manipulator.end_j) == pair


def test_dist_matrix_is_built_on_request():
    coords = random_cloud(12)
    manipulator = CoordManipulator()
    assert manipulator.dist_matrix is None
    manipulator.load_entity(FakeEntity(coords))
    assert np.array_equal(manipulator.dist_matrix, squareform(pdist(coords)))
