"""Fetchers. Only the local-file path runs offline; the rest need the network."""
import shutil

import pytest

from conftest import DATA_DIR, EXPECTED
from crimm.Fetchers import fetch_rcsb


@pytest.fixture
def local_archive(tmp_path):
    """A PDB-archive style directory: <root>/<middle two characters>/<id>.cif."""
    for path in DATA_DIR.glob("*.cif"):
        subdir = tmp_path / path.stem[1:3]
        subdir.mkdir(exist_ok=True)
        shutil.copy(path, subdir / path.name)
    return str(tmp_path)


def test_fetch_from_local_archive(local_archive):
    model = fetch_rcsb("1UBQ", local_entry=local_archive)
    assert model.level == "M"
    assert len(list(model.get_atoms())) == EXPECTED["1ubq"]["n_atoms"]


def test_fetch_from_local_archive_organized(local_archive):
    model = fetch_rcsb("3ptb", local_entry=local_archive, organize=True)
    assert type(model).__name__ == "OrganizedModel"
    assert len(model.ligand) == 1 and len(model.ion) == 1


def test_fetch_missing_local_entry_raises(local_archive):
    with pytest.raises(ValueError, match="Could not load file"):
        fetch_rcsb("9ZZZ", local_entry=local_archive)


def test_ligand_ids_are_rejected():
    with pytest.raises(ValueError, match="not supported"):
        fetch_rcsb("BEN")


@pytest.mark.network
def test_fetch_rcsb_matches_local_fixture():
    model = fetch_rcsb("1CRN")
    found = [(chain.id, chain.chain_type, len(chain)) for chain in model]
    assert found == EXPECTED["1crn"]["chains"]
    assert len(list(model.get_atoms())) == EXPECTED["1crn"]["n_atoms"]


@pytest.mark.network
def test_fetch_rcsb_unknown_id_raises():
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="Could not load file"):
            fetch_rcsb("0XXX")
