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
    # apply_coords rounds each coordinate to 1e-4 A, so the mean is exact to ~1e-5
    assert oriented.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-4)


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


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_orientation_preserves_handedness(method):
    flipped = [
        seed
        for seed in range(N_SEEDS)
        if handedness(orient(method, random_cloud(seed))[1])
        != handedness(random_cloud(seed))
    ]
    assert flipped == []


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_stored_matrix_reproduces_the_orientation(method):
    coords = random_cloud(5)
    manipulator, oriented = orient(method, coords)
    assert manipulator.apply_coords(coords) == pytest.approx(oriented, abs=1e-3)
    assert manipulator.coords == pytest.approx(oriented, abs=1e-3)


@pytest.mark.parametrize("first", ORIENT_METHODS)
@pytest.mark.parametrize("second", ORIENT_METHODS)
def test_stored_matrix_is_the_last_one_applied(first, second):
    coords = random_cloud(9)
    entity = FakeEntity(coords.copy())
    manipulator = CoordManipulator()
    manipulator.load_entity(entity)
    getattr(manipulator, first)()
    once = np.array([atom.coord for atom in entity.atoms])
    getattr(manipulator, second)()
    twice = np.array([atom.coord for atom in entity.atoms])
    assert manipulator.apply_coords(once) == pytest.approx(twice, abs=1e-3)


def test_loading_an_entity_discards_the_previous_matrix():
    manipulator, _ = orient("orient_coords_octa", random_cloud(10))
    coords = random_cloud(11)
    manipulator.load_entity(FakeEntity(coords))
    fresh = CoordManipulator()
    fresh.load_entity(FakeEntity(coords))
    assert manipulator.get_transformation_matrix() == pytest.approx(
        fresh.get_transformation_matrix()
    )


def family(seed):
    """A child entity whose atoms are shared with its parent, which has more."""
    child = FakeEntity(random_cloud(seed))
    parent = FakeEntity(random_cloud(seed + 100, n=20) + 30.0)
    parent.atoms = (
        child.atoms[:30] + parent.atoms[:10] + child.atoms[30:] + parent.atoms[10:]
    )
    child.parent = parent
    return child, parent


def coords_of(entity):
    return np.array([atom.coord for atom in entity.atoms])


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_apply_to_parent_moves_every_parent_atom_once(method):
    child, parent = family(12)
    child_before, parent_before = coords_of(child), coords_of(parent)
    manipulator = CoordManipulator()
    manipulator.load_entity(child)
    getattr(manipulator, method)(apply_to_parent=True)
    # the transform is found from the loaded entity alone ...
    reference, _ = orient(method, child_before)
    assert manipulator.op_mat == pytest.approx(reference.op_mat)
    # ... and applied once to each parent atom, shared ones included
    assert coords_of(parent) == pytest.approx(
        manipulator.apply_coords(parent_before), abs=1e-3
    )
    assert coords_of(child) == pytest.approx(
        manipulator.apply_coords(child_before), abs=1e-3
    )
    # the cached coordinates are those of the moved atoms
    assert manipulator.coords == pytest.approx(coords_of(child), abs=1e-6)
    assert pdist(coords_of(parent)) == pytest.approx(pdist(parent_before), abs=1e-3)


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_without_apply_to_parent_only_the_loaded_entity_moves(method):
    child, parent = family(13)
    child_ids = {id(atom) for atom in child.atoms}
    others_before = [
        atom.coord.copy() for atom in parent.atoms if id(atom) not in child_ids
    ]
    manipulator = CoordManipulator()
    manipulator.load_entity(child)
    getattr(manipulator, method)(apply_to_parent=False)
    others_after = [atom.coord for atom in parent.atoms if id(atom) not in child_ids]
    assert np.array(others_after) == pytest.approx(np.array(others_before))
    assert manipulator.coords == pytest.approx(coords_of(child), abs=1e-6)


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_apply_to_parent_without_parent_warns_and_moves_the_entity(method):
    coords = random_cloud(14)
    entity = FakeEntity(coords.copy())
    manipulator = CoordManipulator()
    manipulator.load_entity(entity)
    with pytest.warns(UserWarning, match="No parent"):
        getattr(manipulator, method)(apply_to_parent=True)
    reference, oriented = orient(method, coords)
    assert coords_of(entity) == pytest.approx(oriented, abs=1e-3)


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_apply_to_parent_on_a_real_structure(load_structure, method):
    structure = load_structure("1ubq")
    model = structure.models[0]
    before = np.array([atom.coord for atom in structure.get_atoms()], dtype=float)
    manipulator = CoordManipulator()
    manipulator.load_entity(model)
    getattr(manipulator, method)(apply_to_parent=True)
    after = np.array([atom.coord for atom in structure.get_atoms()], dtype=float)
    # every atom moved by the stored transform exactly once
    assert after == pytest.approx(manipulator.apply_coords(before), abs=1e-3)
    model_now = np.array([atom.coord for atom in model.get_atoms()], dtype=float)
    assert manipulator.coords == pytest.approx(model_now, abs=1e-6)


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


def chiral_volumes(model):
    """Signed volume of N, C, CB around each CA; positive for L-amino acids."""
    volumes = []
    for residue in model.get_residues():
        if all(name in residue for name in ("N", "CA", "C", "CB")):
            n, ca, c, cb = (residue[name].coord for name in ("N", "CA", "C", "CB"))
            volumes.append(np.dot(np.cross(n - ca, c - ca), cb - ca))
    return np.array(volumes)


@pytest.mark.parametrize("method", ORIENT_METHODS)
def test_orientation_preserves_chiral_centres(load_structure, method):
    model = load_structure("1ubq").models[0]
    before = chiral_volumes(model)
    manipulator = CoordManipulator()
    manipulator.load_entity(model)
    getattr(manipulator, method)()
    after = chiral_volumes(model)
    assert len(before) > 60
    # a proper rotation keeps each signed volume (about 2.5 A^3), sign included
    assert after == pytest.approx(before, abs=1e-2)


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
