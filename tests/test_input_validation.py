"""Validation raises real exceptions, which `python -O` does not strip (#54)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crimm.IO.PRMParser import cmap_field_parser
from crimm.Modeller.CoordManipulator import CoordManipulator
from crimm.Modeller.Solvator import Solvator
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Chain import Ligand

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeAtom:
    def __init__(self, coord):
        self.coord = np.asarray(coord, dtype=float)


class FakeEntity:
    parent = None

    def __init__(self, coords):
        self.atoms = [FakeAtom(coord) for coord in coords]

    def get_atoms(self, include_alt=False):
        return self.atoms


def test_atom_rejects_lower_case_element():
    with pytest.raises(ValueError, match="upper case"):
        Atom("CL", np.zeros(3), element="Cl")


def test_atom_accepts_upper_case_or_missing_element():
    assert Atom("CL", np.zeros(3), element="CL").element == "CL"
    Atom("CA", np.zeros(3), element=None)


def test_cmap_grid_size_mismatch_raises():
    with pytest.raises(ValueError, match="expected 2x2"):
        cmap_field_parser(["1.0 2.0 3.0"], 2)
    assert cmap_field_parser(["1.0 2.0", "3.0 4.0"], 2) == [1.0, 2.0, 3.0, 4.0]


def test_orientation_self_check_raises(monkeypatch):
    """A Kabsch fit that does not reproduce the target axes is an error."""
    from scipy.spatial.transform import Rotation

    import crimm.Modeller.CoordManipulator as cm_module

    class IdentityFit:
        @staticmethod
        def align_vectors(a, b):
            return Rotation.identity(), 0.0

    coords = np.random.default_rng(0).normal(size=(20, 3)) * [6.0, 3.0, 1.5] + 10
    manipulator = CoordManipulator()
    manipulator.load_entity(FakeEntity(coords))
    monkeypatch.setattr(cm_module, "R", IdentityFit)
    with pytest.raises(RuntimeError, match="Kabsch"):
        manipulator.get_transformation_matrix()


def test_solvator_rejects_bad_water_box_shape(organized_model, monkeypatch):
    solvator = Solvator(organized_model("1ubq"))
    monkeypatch.setattr(
        solvator, "get_expelled_water_box_coords", lambda: np.zeros((4, 3))
    )
    with pytest.raises(RuntimeError, match="water box coords shape"):
        solvator._solvate_model()


def _model_with_colliding_next_chain_id(organized_model):
    """Chains sort as strings, so after 'Z' the generated id 'AA' is taken."""
    model = organized_model("3ptb")
    model.add(Ligand("AA"))
    model.add(Ligand("Z"))
    residue = model.ligand[0].residues[0].copy()
    return model, residue


def test_add_ligand_generated_chain_id_collision_raises(organized_model):
    model, residue = _model_with_colliding_next_chain_id(organized_model)
    with pytest.raises(RuntimeError, match="already exists"):
        model.add_ligand(residue)


@pytest.mark.xfail(
    strict=True,
    reason="#167: add_ligand derives the next chain id from the lexically last chain "
    "('Z' -> 'AA') instead of picking a free id",
)
def test_add_ligand_picks_a_free_chain_id(organized_model):
    model, residue = _model_with_colliding_next_chain_id(organized_model)
    n_chains = len(model)
    model.add_ligand(residue)
    assert len(model) == n_chains + 1


def test_validation_survives_python_O():
    """Under -O asserts vanish; the element check must still raise."""
    code = (
        "import sys, numpy as np\n"
        "if not sys.flags.optimize:\n"
        "    sys.exit(3)\n"
        "from crimm.StructEntities.Atom import Atom\n"
        "try:\n"
        "    Atom('CL', np.zeros(3), element='Cl')\n"
        "except ValueError:\n"
        "    print('raised')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "raised"
