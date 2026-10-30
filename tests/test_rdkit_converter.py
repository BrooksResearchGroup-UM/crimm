"""Converting ligand residues to rdkit mols, offline.

A residue read from a CRD file is a plain ``Residue``, not a ``Heterogen``: it has no
``pdbx_description``. Both ways to an rdkit mol have to accept it.
"""

import warnings

import pytest

Chem = pytest.importorskip("rdkit.Chem")

from crimm.Adaptors import RDKitConverter  # noqa: E402
from crimm.IO import CRDParser  # noqa: E402
from crimm.StructEntities.Residue import Heterogen  # noqa: E402

BENZAMIDINE_SMILES = "NC(=N)c1ccccc1"
BENZAMIDINE_CRD = """\
* Benzamidine from 3PTB, heavy atoms
*
         9  EXT
         1         1  BEN       C1             -1.8530000000       14.3110000000       16.6580000000  LIG       1              19.8600000000
         2         1  BEN       C2             -2.1070000000       15.6530000000       16.7580000000  LIG       1              19.8600000000
         3         1  BEN       C3             -1.7740000000       16.3410000000       17.9320000000  LIG       1              19.8600000000
         4         1  BEN       C4             -1.1750000000       15.6620000000       19.0050000000  LIG       1              19.8600000000
         5         1  BEN       C5             -0.9140000000       14.2950000000       18.8850000000  LIG       1              19.8600000000
         6         1  BEN       C6             -1.2570000000       13.6340000000       17.7080000000  LIG       1              19.8600000000
         7         1  BEN       C              -2.1930000000       13.6270000000       15.4960000000  LIG       1              19.8600000000
         8         1  BEN       N1             -2.7970000000       14.2350000000       14.4910000000  LIG       1              19.8600000000
         9         1  BEN       N2             -1.7620000000       12.3910000000       15.3090000000  LIG       1              19.8600000000
"""
# The parts of the CCD entry for BEN that load_heterogen reads, without the hydrogens.
BENZAMIDINE_CCD = {
    "chem_comp_atom": {
        "atom_id": ["C1", "C2", "C3", "C4", "C5", "C6", "C", "N1", "N2"],
        "type_symbol": ["C", "C", "C", "C", "C", "C", "C", "N", "N"],
        "charge": [0] * 9,
    },
    "chem_comp_bond": {
        "atom_id_1": ["C1", "C1", "C2", "C3", "C4", "C5", "C1", "C", "C"],
        "atom_id_2": ["C2", "C6", "C3", "C4", "C5", "C6", "C", "N1", "N2"],
        "value_order": ["AROM"] * 6 + ["SING", "DOUB", "SING"],
        "pdbx_aromatic_flag": ["Y"] * 6 + ["N"] * 3,
    },
}


@pytest.fixture
def crd_residue(tmp_path):
    """Benzamidine as a plain ``Residue``, read from a CRD file."""
    path = tmp_path / "ben.crd"
    path.write_text(BENZAMIDINE_CRD)
    structure = CRDParser(QUIET=True).get_structure(str(path))
    residue = structure.models[0].chains[0].residues[0]
    assert not isinstance(residue, Heterogen)
    assert not hasattr(residue, "pdbx_description")
    return residue


def test_heterogen_to_rdkit_accepts_crd_residue(crd_residue):
    mol = RDKitConverter.heterogen_to_rdkit(crd_residue, smiles=BENZAMIDINE_SMILES)
    assert mol is not None
    assert mol.GetNumHeavyAtoms() == 9
    assert sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) == 6
    assert not mol.HasProp("Description")


def test_heterogen_to_rdkit_keeps_description_of_heterogen(organized_model):
    ligand = organized_model("3ptb").ligand[0].residues[0]
    assert isinstance(ligand, Heterogen)
    mol = RDKitConverter.heterogen_to_rdkit(ligand, smiles=BENZAMIDINE_SMILES)
    assert mol.GetProp("Description") == "BENZAMIDINE"


def test_load_heterogen_warns_on_crd_residue_and_converts(crd_residue, monkeypatch):
    monkeypatch.setattr(
        RDKitConverter, "fetch_rcsb_as_dict", lambda resname: BENZAMIDINE_CCD
    )
    converter = RDKitConverter.RDKitHetConverter()
    with pytest.warns(UserWarning, match="not a Heterogen"):
        converter.load_heterogen(crd_residue, update_hydrogens=False)
    assert converter.lig is crd_residue
    assert converter.get_mol().GetNumHeavyAtoms() == 9


def test_load_heterogen_does_not_warn_on_heterogen(organized_model, monkeypatch):
    monkeypatch.setattr(
        RDKitConverter, "fetch_rcsb_as_dict", lambda resname: BENZAMIDINE_CCD
    )
    ligand = organized_model("3ptb").ligand[0].residues[0]
    converter = RDKitConverter.RDKitHetConverter()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not a Heterogen.*")
        converter.load_heterogen(ligand, update_hydrogens=False)
    assert converter.get_mol().GetNumHeavyAtoms() == 9
