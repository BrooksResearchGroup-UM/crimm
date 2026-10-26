"""Pure-function tests for crimm.Utils.StructureUtils (no structures needed)."""
import numpy as np
import pytest

from crimm.Utils.StructureUtils import (
    compact_charmm_chain_id,
    get_coords,
    index_to_letters,
    letters_to_index,
    polymer_chain_id_to_charmm_segid,
)


@pytest.mark.parametrize(
    "index, letters",
    [(0, "A"), (25, "Z"), (26, "AA"), (701, "ZZ"), (702, "AAA")],
)
def test_index_letters_documented_examples(index, letters):
    assert index_to_letters(index) == letters
    assert letters_to_index(letters) == index


def test_index_letters_round_trip():
    for index in range(2000):
        assert letters_to_index(index_to_letters(index)) == index


@pytest.mark.parametrize(
    "chain_type, chain_id, segid",
    [
        ("Polypeptide(L)", "A", "PROA"),
        ("Polypeptide(L)", "AA", "PRAA"),
        ("Polyribonucleotide", "A", "RNAA"),
        ("Polyribonucleotide", "AA", "RRAA"),
        ("Polydeoxyribonucleotide", "A", "DNAA"),
        ("Polydeoxyribonucleotide", "AA", "DRAA"),
    ],
)
def test_polymer_segid_documented_rules(chain_type, chain_id, segid):
    assert polymer_chain_id_to_charmm_segid(chain_type, chain_id) == segid


@pytest.mark.parametrize("chain_id", ["A", "AA", "AAA", "ZZZ", "ABCD"])
def test_polymer_segid_fits_charmm_limit(chain_id):
    segid = polymer_chain_id_to_charmm_segid("Polypeptide(L)", chain_id)
    assert 1 <= len(segid) <= 4
    assert segid.isalnum()


def test_compact_chain_id_keeps_short_ids():
    assert compact_charmm_chain_id("a") == "A"
    assert compact_charmm_chain_id("AB1") == "AB1"


def test_compact_chain_id_rejects_unrepresentable():
    with pytest.raises(ValueError):
        compact_charmm_chain_id("A-B-C")


def test_get_coords_shapes(load_structure):
    model = load_structure("1crn").models[0]
    coords = get_coords(model)
    assert coords.shape == (327, 3)
    assert np.isfinite(coords).all()
    chain = model["A"]
    assert get_coords([chain, chain]).shape == (654, 3)
    atom = next(chain.get_atoms())
    assert get_coords(atom).shape == (3,)


def test_get_coords_rejects_non_entity():
    with pytest.raises(TypeError):
        get_coords("not an entity")
