"""Renumbering a residue with alternative residue names (a point mutation stored
as a ``DisorderedResidue``) must move every alternative with it (#153).

Crambin (1CRN) stands in for a structure with sequence heterogeneity: Pro22 gets a
Ser22 alternative, built the way ``StructureBuilder`` builds one from two residues
that share a position.
"""

import warnings

import pytest
from conftest import parse_structure

from crimm.IO.StructureBuilder import StructureBuilder
from crimm.StructEntities.Residue import DisorderedResidue
from crimm.Utils.StructureUtils import merge_chains, renumber_by_sequence

CRN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def _with_point_mutation(resseq=22, alt_resname="SER"):
    """1CRN chain A where residue ``resseq`` is a DisorderedResidue holding the
    deposited residue (altloc A) and a copy renamed ``alt_resname`` (altloc B)."""
    chain = parse_structure("1crn").models[0]["A"]
    res = chain[resseq]
    alt = res.copy()
    alt.resname = alt_resname
    for atom in res:
        atom.altloc = "A"
    for atom in alt:
        atom.altloc = "B"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        StructureBuilder()._process_duplicated_res(alt, res, chain)
    chain.sort_residues()
    chain.reset_disordered_residues()
    (disordered,) = chain.get_disordered_res()
    assert disordered.id == (" ", resseq, " ")
    assert sorted(disordered.child_dict) == sorted([res.resname, alt_resname])
    return chain


def _alternative_ids(residue):
    return {name: child.id for name, child in residue.child_dict.items()}


def test_fixture_has_no_missing_residues():
    chain = _with_point_mutation()
    assert chain.missing_res == []


def test_setting_the_id_moves_every_alternative():
    chain = _with_point_mutation()
    disordered = chain[22]
    disordered.id = (" ", 122, " ")

    assert _alternative_ids(disordered) == {
        "PRO": (" ", 122, " "),
        "SER": (" ", 122, " "),
    }
    # The chain finds the residue under its new number only.
    assert chain.child_dict[(" ", 122, " ")] is disordered
    assert (" ", 22, " ") not in chain.child_dict
    # Cached full ids are rebuilt.
    for child in disordered.child_dict.values():
        assert child.get_full_id()[-1] == (" ", 122, " ")
        assert all(atom.get_full_id()[3] == (" ", 122, " ") for atom in child)


def test_copy_keeps_the_new_number():
    chain = _with_point_mutation()
    disordered = chain[22]
    disordered.id = (" ", 122, " ")
    new = disordered.copy()
    assert new.id == (" ", 122, " ")
    assert set(_alternative_ids(new).values()) == {(" ", 122, " ")}


def test_constructor_still_sets_the_id():
    res = DisorderedResidue((" ", 5, " "))
    assert res.id == res.get_id() == (" ", 5, " ")
    assert res.parent is None


def test_renumber_by_sequence_moves_the_alternatives():
    chain = _with_point_mutation()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        renumbered = renumber_by_sequence(chain, "GSHM" + CRN_SEQ)

    (disordered,) = renumbered.get_disordered_res()
    assert disordered.id == (" ", 26, " ")
    assert set(_alternative_ids(disordered).values()) == {(" ", 26, " ")}
    # Only the four added N-terminal residues are missing: position 26 is present
    # and no old number (22) is reported as present.
    assert [resseq for resseq, _ in renumbered.missing_res] == [1, 2, 3, 4]


def test_merge_chains_moves_the_alternatives():
    chain = _with_point_mutation()
    # Number the input away from 1..N, so merge_chains has to renumber.
    for res in reversed(list(chain)):
        res.id = (res.id[0], res.id[1] + 100, res.id[2])
    assert set(_alternative_ids(chain[122]).values()) == {(" ", 122, " ")}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = merge_chains([chain], new_chain_id="X")

    (disordered,) = merged.get_disordered_res()
    assert disordered.id == (" ", 22, " ")
    assert set(_alternative_ids(disordered).values()) == {(" ", 22, " ")}
    assert merged.missing_res == []


@pytest.mark.parametrize("new_id", [(" ", 22, " "), (" ", 23, " ")])
def test_setting_an_id_in_use_warns_like_a_plain_residue(new_id):
    """Same contract as ``Entity.id``: an id already used by a sibling warns.
    Setting the current id again is a no-op and does not warn."""
    chain = _with_point_mutation()
    disordered = chain[22]
    if new_id == disordered.id:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            disordered.id = new_id
        assert chain.child_dict[new_id] is disordered
    else:
        with pytest.warns(Warning, match="already used"):
            disordered.id = new_id
