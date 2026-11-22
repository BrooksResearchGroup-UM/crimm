"""crimm.Utils.query_db.find_uniprot_ids: RCSB requests are stubbed except in the
``network`` test."""

import warnings

import pytest

from crimm.Utils import query_db
from crimm.Utils.query_db import find_uniprot_ids

UBQ_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
UBQ_VARIANT = UBQ_SEQ[:10] + "A" + UBQ_SEQ[11:]


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _instance(rcsb_id, seq, uniprot_ids):
    return {
        "rcsb_id": rcsb_id,
        "polymer_entity": {
            "entity_poly": {"pdbx_seq_one_letter_code_can": seq},
            "rcsb_polymer_entity_container_identifiers": {"uniprot_ids": uniprot_ids},
        },
    }


INSTANCES = [
    _instance("1UBQ.A", UBQ_SEQ, ["P0CG48"]),
    _instance("9XXX.A", UBQ_VARIANT, ["P0CG47"]),
    _instance("9XXX.B", UBQ_SEQ, ["P0CG47"]),
    _instance("8YYY.C", UBQ_SEQ[:38], None),
]


@pytest.fixture
def stub_rcsb(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout, proxies=None):
        if url == query_db._RCSB_SEARCH_URL:
            sent["search"] = json
            hits = [{"identifier": i["rcsb_id"], "score": 1.0} for i in INSTANCES]
            return _Response(200, {"result_set": hits})
        sent["graphql"] = json
        return _Response(200, {"data": {"polymer_entity_instances": INSTANCES}})

    monkeypatch.setattr(query_db.requests, "post", fake_post)
    return sent


def test_groups_hits_by_accession(stub_rcsb):
    result = find_uniprot_ids(" mqifvktltg" + UBQ_SEQ[10:].lower() + "\n")
    assert [g["uniprot_id"] for g in result] == ["P0CG47", "P0CG48", None]
    p47, p48, unmapped = result
    assert p47["n_chains"] == 2 and p47["chains"] == ["9XXX.A", "9XXX.B"]
    assert p47["best_identity"] == 1.0 and p47["best_coverage"] == 1.0
    assert p48["chains"] == ["1UBQ.A"]
    assert unmapped["chains"] == ["8YYY.C"]
    assert unmapped["best_coverage"] == pytest.approx(38 / 76, abs=1e-3)
    assert stub_rcsb["search"]["query"]["parameters"]["value"] == UBQ_SEQ
    assert stub_rcsb["graphql"]["variables"]["ids"] == [
        "1UBQ.A",
        "9XXX.A",
        "9XXX.B",
        "8YYY.C",
    ]


def test_identity_of_a_variant():
    identity, coverage = query_db._local_identity(UBQ_SEQ, UBQ_VARIANT)
    assert identity == pytest.approx(75 / 76, abs=1e-3)
    assert coverage == 1.0


def test_identity_with_selenocysteine_and_pyrrolysine():
    # U and O are not in BLOSUM62; they must still align and count as identical
    seq = UBQ_SEQ[:20] + "U" + UBQ_SEQ[21:50] + "O" + UBQ_SEQ[51:]
    assert query_db._local_identity(seq, seq) == (1.0, 1.0)
    identity, coverage = query_db._local_identity(seq, UBQ_SEQ)
    assert coverage == 1.0
    assert identity == round(74 / 76, 3)


def test_accepts_a_chain(stub_rcsb, load_structure):
    chain = load_structure("1ubq", include_solvent=False).models[0]["A"]
    find_uniprot_ids(chain, identity_cutoff=0.5, max_hits=7)
    search = stub_rcsb["search"]
    assert search["query"]["parameters"]["value"] == UBQ_SEQ
    assert search["query"]["parameters"]["identity_cutoff"] == 0.5
    assert search["request_options"]["paginate"]["rows"] == 7


def test_chain_without_sequence_raises(load_structure):
    chain = load_structure("1ubq", include_solvent=False).models[0]["A"]
    chain.can_seq = ""
    with pytest.raises(ValueError, match="no canonical sequence"):
        find_uniprot_ids(chain)


def test_rejects_non_sequence():
    with pytest.raises(ValueError, match="Not a one-letter sequence"):
        find_uniprot_ids("MQI-FV")


def test_no_match_gives_empty_list(monkeypatch):
    monkeypatch.setattr(
        query_db.requests,
        "post",
        lambda url, json, timeout, proxies=None: _Response(204),
    )
    assert find_uniprot_ids(UBQ_SEQ) == []


def test_http_error_warns_and_returns_none(monkeypatch):
    monkeypatch.setattr(
        query_db.requests,
        "post",
        lambda url, json, timeout, proxies=None: _Response(500),
    )
    with pytest.warns(UserWarning, match="Status Code"):
        assert find_uniprot_ids(UBQ_SEQ) is None


@pytest.mark.network
def test_ubiquitin_maps_to_p0cg48():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = find_uniprot_ids(UBQ_SEQ, identity_cutoff=0.95, max_hits=100)
    accessions = {g["uniprot_id"]: g for g in result}
    assert "P0CG48" in accessions
    assert accessions["P0CG48"]["best_identity"] == 1.0
