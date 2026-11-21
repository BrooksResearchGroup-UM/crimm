"""Loop building from user-supplied inputs: PDB-only chains, a canonical sequence,
a template structure and a UniProt ID.

The model case is a single-chain construct deposited as two chains without sequence
records, e.g. a Dengue NS2B-NS3 protease with the cofactor and protease in separate
chains and the linker unresolved. Ubiquitin (1UBQ) stands in for it here: chain A is
cut into renumbered pieces across two chains, written as a PDB file and read back.
"""

import re
import warnings

import numpy as np
import pytest
from Bio.Seq import Seq
from conftest import DATA_DIR

from crimm.IO import MMCIFParser, PDBParser
from crimm.IO.PDBString import get_pdb_str
from crimm.Modeller import LoopBuilder
from crimm.Modeller.LoopBuilder import ChainLoopBuilder
from crimm.Utils.StructureUtils import merge_chains, renumber_by_sequence

UBQ_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture
def ubq_chain(load_structure):
    return load_structure("1ubq", include_solvent=False).models[0]["A"]


def _pieces_pdb(ubq_chain, tmp_path, b_offset=0, mutate=None):
    """PDB file with ubiquitin 1-30 as chain A (numbered 101-130) and 41-49 plus
    53-76 as chain B. Residues 31-40 and 50-52 are missing. ``mutate`` maps a
    residue number to a new residue name, applied to the name only."""
    lines = []
    for line in get_pdb_str(ubq_chain).splitlines():
        if not line.startswith("ATOM"):
            continue
        resseq = int(line[22:26])
        resname = line[17:20]
        if mutate and resseq in mutate:
            resname = mutate[resseq]
        if resseq <= 30:
            chain, new = "A", resseq + 100
        elif 41 <= resseq <= 49:
            chain, new = "B", resseq
        elif resseq >= 53:
            chain, new = "B", resseq + b_offset
        else:
            continue
        lines.append(f"{line[:17]}{resname}{line[20]}{chain}{new:4d}{line[26:]}")
    path = tmp_path / "pieces.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDBParser(QUIET=True).get_structure(str(path)).models[0]


def _merged(model, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return merge_chains([model["A"], model["B"]], sequence=UBQ_SEQ, **kwargs)


# --- parser -------------------------------------------------------------------


def _strip_one_letter_codes(text):
    """Remove _entity_poly.pdbx_seq_one_letter_code(_can) items, as in AlphaFold
    Server output, including multi-line ;-delimited values."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if re.match(r"_entity_poly\.pdbx_seq_one_letter_code(_can)?\s*$", lines[i]):
            i += 1
            assert lines[i].startswith(";")
            i += 1
            while not lines[i].startswith(";"):
                i += 1
            i += 1
            continue
        if re.match(r"_entity_poly\.pdbx_seq_one_letter_code(_can)?\s", lines[i]):
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + "\n"


def test_mmcif_without_one_letter_codes_gets_sequence_from_entity_poly_seq(tmp_path):
    text = (DATA_DIR / "1ubq.cif").read_text()
    stripped = _strip_one_letter_codes(text)
    assert "_entity_poly.pdbx_seq_one_letter_code" not in stripped
    path = tmp_path / "no_one_letter.cif"
    path.write_text(stripped)
    chain = MMCIFParser(QUIET=True).get_structure(str(path)).models[0]["A"]
    assert str(chain.can_seq) == UBQ_SEQ
    assert len(chain.missing_res) == 0


# --- merge_chains / renumber_by_sequence ----------------------------------------


def test_pdb_chains_have_no_sequence(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path)
    assert not model["A"].can_seq
    assert [r.id[1] for r in model["A"]][:2] == [101, 102]


def test_merge_without_sequence_concatenates_in_order(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path)
    merged = merge_chains([model["A"], model["B"]], new_chain_id="X")
    assert merged.id == "X"
    assert len(merged) == 30 + 9 + 24
    assert [r.id[1] for r in merged] == list(range(1, 64))
    assert merged[31].xtra["source_id"] == ("B", (" ", 41, " "))
    # inputs untouched
    assert model["A"].child_list[0].id[1] == 101


def test_merge_with_sequence_numbers_by_sequence_and_finds_gaps(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path)
    chain = _merged(model)
    assert [(min(g), max(g)) for g in chain.gaps] == [(31, 40), (50, 52)]
    assert str(chain.can_seq) == UBQ_SEQ
    assert [(m["start"], m["end"]) for m in chain.sequence_mapping] == [
        (1, 30),
        (41, 49),
        (53, 76),
    ]
    assert chain.sequence_mapping[0]["source_first"] == "A:101"
    for res in chain:
        assert res.resname == ubq_chain[res.id[1]].resname
        assert res["CA"].coord == pytest.approx(ubq_chain[res.id[1]]["CA"].coord)


def test_mismatch_kept_by_default_and_not_a_gap(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path, mutate={45: "TRP"})
    with pytest.warns(UserWarning, match=r"TRP45 \(sequence: PHE\)"):
        chain = merge_chains([model["A"], model["B"]], sequence=UBQ_SEQ)
    assert chain[45].resname == "TRP"
    assert 45 not in {i for g in chain.gaps for i in g}
    assert chain.can_seq[44] == "W"
    assert chain.sequence_mapping[1]["mismatches"] == [(45, "TRP", "PHE")]


def test_mismatch_substituted_on_request(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path, mutate={45: "TRP"})
    with pytest.warns(UserWarning, match="replaced by the sequence's residue"):
        chain = merge_chains(
            [model["A"], model["B"]], sequence=UBQ_SEQ, substitute_mismatches=True
        )
    assert chain[45].resname == "PHE"
    assert {a.get_id() for a in chain[45]} == {"N", "CA", "C", "O", "CB"}
    assert chain.can_seq[44] == "F"


def test_inconsistent_offsets_within_a_chain_warn(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path, b_offset=100)
    with pytest.warns(UserWarning, match="different numbering offsets"):
        merge_chains([model["A"], model["B"]], sequence=UBQ_SEQ)


def test_piece_without_a_match_raises(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path)
    with pytest.raises(ValueError, match="matches no stretch"):
        merge_chains([model["A"], model["B"]], sequence=UBQ_SEQ[:60] + "A" * 16)


def test_ambiguous_placement_raises(ubq_chain, tmp_path):
    model = _pieces_pdb(ubq_chain, tmp_path)
    piece = model["B"]
    # the sequence holds residues 41-76 twice, so chain B fits in two places
    doubled = UBQ_SEQ + "GGGG" + UBQ_SEQ[40:]
    with pytest.raises(ValueError, match="not unique"):
        renumber_by_sequence(piece, doubled)


def test_renumber_rejects_nucleic_acids(load_structure):
    chain = load_structure("1bna").models[0]["A"]
    with pytest.raises(NotImplementedError):
        renumber_by_sequence(chain, "CGCGAATTCGCG")


# --- ChainLoopBuilder with a user template -------------------------------------


def _full_template(load_structure):
    return load_structure("1ubq", include_solvent=False).models[0]["A"]


def test_build_from_template_fills_gaps_and_reports(
    ubq_chain, tmp_path, load_structure
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    builder = ChainLoopBuilder(chain, uniprot_id="P0CG48")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_template(_full_template(load_structure))
    out = builder.get_chain()
    assert len(out) == 76
    assert len(out.missing_res) == 0
    assert [item["gap"] for item in builder.build_report] == [(31, 40), (50, 52)]
    for item in builder.build_report:
        assert item["local_rmsd"] < 0.01
        for dist in item["junction_CN"].values():
            assert dist == pytest.approx(1.33, abs=0.05)


def test_build_from_template_file_path(ubq_chain, tmp_path):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_template(DATA_DIR / "1ubq.cif")
    assert len(builder.get_chain()) == 76
    assert builder.build_report[0]["bfactor_label"] == "B-factor"


def test_template_pdb_path_is_refused(ubq_chain, tmp_path):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with pytest.raises(ValueError, match="only mmCIF templates"):
        builder.build_from_template(tmp_path / "pieces.pdb")


def test_template_with_different_residues_in_gap_is_skipped(
    ubq_chain, tmp_path, load_structure
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    # a variant template: G35A in both its sequence and its coordinates
    template = _full_template(load_structure)
    template[35].resname = "ALA"
    template.reported_res[34] = (35, "ALA")
    template.can_seq = Seq(UBQ_SEQ[:34] + "A" + UBQ_SEQ[35:])
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with pytest.warns(
        UserWarning, match=r"Gap 31-40 not built.*GLY35 \(template: ALA\)"
    ):
        builder.build_from_template(template)
    assert [item["gap"] for item in builder.build_report] == [(50, 52)]

    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_template(template, allow_mismatch=True)
    assert len(builder.get_chain()) == 76
    assert builder.build_report[0]["name_mismatches"] == [(35, "GLY", "ALA")]


def test_terminal_gaps_only_on_request(ubq_chain, tmp_path, load_structure):
    model = _pieces_pdb(ubq_chain, tmp_path)
    chain = _merged(model)
    for res_id in (1, 2, 76):
        chain.detach_child((" ", res_id, " "))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_template(_full_template(load_structure))
    assert [(min(g), max(g)) for g in builder.model_chain.gaps] == [(1, 2), (76, 76)]

    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_template(
            _full_template(load_structure), include_terminal=True
        )
    assert len(builder.get_chain().missing_res) == 0


def test_stretched_junction_warns(ubq_chain, tmp_path, load_structure):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    template = _full_template(load_structure)
    for res_id in range(31, 41):
        for atom in template[res_id]:
            atom.coord = atom.coord + np.array([0.0, 0.0, 6.0])
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with pytest.warns(UserWarning, match=r"do not join the chain.*31-40 \(before"):
        builder.build_from_template(template)


# --- ChainLoopBuilder with a UniProt ID ------------------------------------------


def test_build_from_alphafold_uses_given_uniprot_id(
    ubq_chain, tmp_path, load_structure, monkeypatch
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    template_structure = load_structure("1ubq", include_solvent=False)
    requested = []

    def fake_fetch(uniprot_id):
        requested.append(uniprot_id)
        return template_structure

    def no_lookup(*args):
        raise AssertionError("PDB lookup must not run when a UniProt ID is given")

    monkeypatch.setattr(LoopBuilder, "fetch_alphafold", fake_fetch)
    monkeypatch.setattr(LoopBuilder, "uniprot_id_query", no_lookup)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = ChainLoopBuilder(chain, uniprot_id="P0CG48")
        builder.build_from_alphafold()
    assert requested == ["P0CG48"]
    assert len(builder.get_chain()) == 76
    assert builder.build_report[0]["bfactor_label"] == "pLDDT"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = ChainLoopBuilder(chain, pdbid="NONE")
        builder.build_from_alphafold(uniprot_id="P62988")
    assert requested[-1] == "P62988"


def test_no_pdb_id_warning_when_uniprot_id_given(ubq_chain, tmp_path):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ChainLoopBuilder(chain, uniprot_id="P0CG48")
    assert not [w for w in caught if "PDB ID" in str(w.message)]


# --- ChainLoopBuilder with PDB templates: search sequence and chosen entries -----


def _fake_fetch_rcsb(load_structure, calls=None):
    def fetch(pdb_id, **kwargs):
        if calls is not None:
            calls.append(pdb_id)
        if pdb_id != "1UBQ":
            raise ValueError(f"Could not load file for {pdb_id}")
        return load_structure("1ubq", include_solvent=False).models[0]

    return fetch


class _FakeResponse:
    status_code = 200
    text = '{"result_set": [{"identifier": "1UBQ.A", "score": 1.0}]}'


class _NoContent:
    status_code = 204
    text = ""


def test_query_seq_match_uses_given_sequence(ubq_chain, tmp_path, monkeypatch):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json)
        return _FakeResponse()

    monkeypatch.setattr(LoopBuilder.requests, "post", fake_post)
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    assert builder.query_seq_match(5, 0.9) == {1.0: {"1UBQ": ["A"]}}
    assert sent[-1]["query"]["parameters"]["value"] == UBQ_SEQ
    builder.query_seq_match(5, 0.9, sequence=" mqifvktltg\nktitlev ")
    assert sent[-1]["query"]["parameters"]["value"] == "MQIFVKTLTGKTITLEV"
    with pytest.raises(ValueError, match="Not a one-letter sequence"):
        builder.query_seq_match(5, 0.9, sequence="MQI-FV")
    assert sent[-1]["query"]["type"] == "terminal"


def test_uniprot_filter_is_added_to_the_search(ubq_chain, tmp_path, monkeypatch):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json)
        return _FakeResponse()

    monkeypatch.setattr(LoopBuilder.requests, "post", fake_post)
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    builder.query_seq_match(5, 0.9, uniprot_id="p0cg48")
    query = sent[-1]["query"]
    assert query["type"] == "group" and query["logical_operator"] == "and"
    sequence_node, accession_node = query["nodes"]
    assert sequence_node["parameters"]["value"] == UBQ_SEQ
    assert accession_node["parameters"]["value"] == "P0CG48"
    assert accession_node["parameters"]["attribute"].endswith("database_accession")


def test_empty_uniprot_filtered_search_warns_with_hint(
    ubq_chain, tmp_path, monkeypatch
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    monkeypatch.setattr(
        LoopBuilder.requests, "post", lambda url, json, timeout: _NoContent()
    )
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with pytest.warns(UserWarning, match="strain- or isoform-specific accession"):
        builder.build_from_homology(uniprot_id="P14340")
    assert builder.query_results == {}


def test_build_from_homology_passes_sequence_and_fills_gaps(
    ubq_chain, tmp_path, load_structure, monkeypatch
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    queried = []

    def fake_query(
        self, max_num_match, identity_score_cutoff, sequence=None, uniprot_id=None
    ):
        queried.append(sequence)
        return {1.0: {"1UBQ": ["A"]}}

    monkeypatch.setattr(ChainLoopBuilder, "query_seq_match", fake_query)
    monkeypatch.setattr(LoopBuilder, "fetch_rcsb", _fake_fetch_rcsb(load_structure))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_homology(sequence="MQIFVKTLTGK")
    assert queried == ["MQIFVKTLTGK"]
    assert len(builder.get_chain()) == 76
    assert [item["template"] for item in builder.build_report] == ["1UBQ.A"] * 2


@pytest.mark.parametrize(
    "pdbids",
    ["1ubq", "1UBQ.A", ["1UBQ_A"], {"1UBQ": ["A"]}, {"1UBQ": None}],
)
def test_build_from_pdbids_fills_gaps(
    ubq_chain, tmp_path, load_structure, monkeypatch, pdbids
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    monkeypatch.setattr(LoopBuilder, "fetch_rcsb", _fake_fetch_rcsb(load_structure))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.build_from_pdbids(pdbids)
    assert len(builder.get_chain()) == 76
    assert [item["gap"] for item in builder.build_report] == [(31, 40), (50, 52)]
    assert builder.build_report[0]["template"] == "1UBQ.A"
    assert builder.build_report[0]["bfactor_label"] == "B-factor"


def test_build_from_pdbids_skips_bad_entries_and_chains(
    ubq_chain, tmp_path, load_structure, monkeypatch
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    calls = []
    monkeypatch.setattr(
        LoopBuilder, "fetch_rcsb", _fake_fetch_rcsb(load_structure, calls)
    )
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builder.build_from_pdbids(["9ZZZ", "1UBQ.Q", "1UBQ.A"])
    messages = [str(w.message) for w in caught]
    assert any("9ZZZ skipped" in m for m in messages)
    assert any("has no chain Q" in m for m in messages)
    assert calls == ["9ZZZ", "1UBQ"]
    assert len(builder.get_chain()) == 76


def test_build_from_pdbids_rejects_malformed_id(ubq_chain, tmp_path):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with pytest.raises(ValueError, match="Not a PDB ID"):
        builder.build_from_pdbids(["UBQ"])


def test_pdb_templates_with_different_residues_leave_gap_unbuilt(
    ubq_chain, tmp_path, load_structure, monkeypatch
):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))

    def fetch_variant(pdb_id, **kwargs):
        model = load_structure("1ubq", include_solvent=False).models[0]
        template = model["A"]
        template[35].resname = "ALA"
        template.reported_res[34] = (35, "ALA")
        template.can_seq = Seq(UBQ_SEQ[:34] + "A" + UBQ_SEQ[35:])
        return model

    monkeypatch.setattr(LoopBuilder, "fetch_rcsb", fetch_variant)
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    with pytest.warns(UserWarning, match=r"Gap 31-40 not built.*every template"):
        builder.build_from_pdbids("1UBQ")
    assert [item["gap"] for item in builder.build_report] == [(50, 52)]


@pytest.mark.network
def test_rcsb_sequence_search_finds_ubiquitin(ubq_chain, tmp_path):
    chain = _merged(_pieces_pdb(ubq_chain, tmp_path))
    builder = ChainLoopBuilder(chain, pdbid="NONE")
    results = builder.query_seq_match(50, 0.95, sequence=UBQ_SEQ)
    assert any("1UBQ" in entries for entries in results.values())
    # 1UBQ is mapped to P0CG48 (polyubiquitin-C), so the filter keeps it ...
    results = builder.query_seq_match(50, 0.95, uniprot_id="P0CG48")
    assert any("1UBQ" in entries for entries in results.values())
    # ... and an unrelated accession leaves nothing
    assert builder.query_seq_match(50, 0.95, uniprot_id="P69905") == {}


# --- review fixes -----------------------------------------------------------------


def test_merge_keeps_dna_sequence(load_structure):
    strand = load_structure("1bna", include_solvent=False).models[0]["A"]
    merged = merge_chains([strand])
    assert str(merged.can_seq) == "CGCGAATTCGCG"


@pytest.mark.parametrize("resseq, resname", [(1, "MSE"), (68, "HSE")])
def test_variant_name_matching_the_sequence_is_not_a_gap(
    ubq_chain, tmp_path, resseq, resname
):
    model = _pieces_pdb(ubq_chain, tmp_path, mutate={resseq: resname})
    chain = _merged(model)
    assert chain[resseq].resname == resname
    assert [(min(g), max(g)) for g in chain.gaps] == [(31, 40), (50, 52)]

