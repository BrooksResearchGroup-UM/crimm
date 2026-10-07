"""CoordManipulator: orientation must be a rigid, proper transformation."""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

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
                reason="No determinant check on the PCA axes: about half of all inputs "
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
                    reason="PCA methods store op_mat in a row-vector layout that "
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
    reason="box_dim calls ndarray.ptp, which NumPy 2 removed",
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
