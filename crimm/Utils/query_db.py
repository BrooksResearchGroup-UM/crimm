import warnings

import pandas as pd
import requests


def uniprot_id_query(pdbid, entity_id, proxies=None):
    """Query wth the RCSB PDB API for uniprot id for a given pdb id and
    chain entity id
    """
    rcsb_url = f"https://data.rcsb.org/rest/v1/core/uniprot/{pdbid}/{entity_id}"
    response = requests.get(rcsb_url, timeout=500, proxies=proxies)
    if (code := response.status_code) != 200:
        warnings.warn(
            f'GET request on RCSB for "{pdbid}-{entity_id}" for uniprot ID '
            f"did not return valid result! \n[Status Code] {code}"
        )
        return
    kw1 = "rcsb_uniprot_container_identifiers"
    kw2 = "uniprot_id"
    uniprot_id = response.json()[0][kw1][kw2]
    return uniprot_id


_RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
_INSTANCE_QUERY = """
query($ids: [String!]!) {
  polymer_entity_instances(instance_ids: $ids) {
    rcsb_id
    polymer_entity {
      entity_poly { pdbx_seq_one_letter_code_can }
      rcsb_polymer_entity_container_identifiers { uniprot_ids }
    }
  }
}
"""


def _local_identity(query, target):
    """(identity, coverage) of the best local alignment of target on query:
    identical / aligned residues, and aligned residues / len(query)."""
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query, target)[0]
    identical = aligned = 0
    for (q0, q1), (t0, t1) in zip(*alignment.aligned):
        aligned += q1 - q0
        identical += sum(a == b for a, b in zip(query[q0:q1], target[t0:t1]))
    if aligned == 0:
        return 0.0, 0.0
    return round(identical / aligned, 3), round(aligned / len(query), 3)


def find_uniprot_ids(sequence, identity_cutoff=0.9, max_hits=50, proxies=None):
    """Find the UniProt accessions the PDB uses for structures of a sequence.

    The PDB maps each entity to the accession its depositors used, often a
    strain- or isoform-specific entry rather than the reference one (e.g.
    Dengue-2 NS2B-NS3 protease entries are mapped to Q91H74, P12823 or
    P14337, not to the reference polyprotein P14340). This runs the RCSB
    sequence search and reports the accessions of the chains it finds.

    Args:
        sequence: one-letter sequence, or a PolymerChain (its canonical
            sequence is used).
        identity_cutoff: lowest sequence identity of a hit (0 to 1), passed
            to the RCSB search.
        max_hits: number of chains (polymer instances) to retrieve.
        proxies: proxies for ``requests``.

    Returns:
        A list with one dict per accession, most chains first, then highest
        identity: ``uniprot_id``, ``n_chains``, ``best_identity``,
        ``best_coverage`` (fraction of the query aligned) and ``chains``
        (RCSB instance IDs such as "2FOM.B"). Chains without a UniProt
        mapping come last with ``uniprot_id`` None. Identity and coverage are
        computed locally from a BLOSUM62 local alignment. Returns ``[]`` if
        nothing matches and None (with a warning) if a request fails.
    """
    if hasattr(sequence, "can_seq"):
        if not sequence.can_seq:
            raise ValueError(
                f"{sequence} has no canonical sequence (e.g. it was read from a "
                "PDB file); give the sequence instead."
            )
        sequence = sequence.can_seq
    query_seq = "".join(str(sequence).split()).upper()
    if not query_seq.isalpha():
        raise ValueError(f"Not a one-letter sequence: {sequence!r}")

    search = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": query_seq,
            },
        },
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {"start": 0, "rows": max_hits},
        },
        "return_type": "polymer_instance",
    }
    response = requests.post(
        _RCSB_SEARCH_URL, json=search, timeout=500, proxies=proxies
    )
    if response.status_code == 204:  # RCSB: no match
        return []
    if response.status_code != 200:
        warnings.warn(
            "RCSB sequence search did not return a valid result! "
            f"\n[Status Code] {response.status_code}"
        )
        return None
    instance_ids = [hit["identifier"] for hit in response.json()["result_set"]]

    response = requests.post(
        _RCSB_GRAPHQL_URL,
        json={"query": _INSTANCE_QUERY, "variables": {"ids": instance_ids}},
        timeout=500,
        proxies=proxies,
    )
    if response.status_code != 200:
        warnings.warn(
            "RCSB GraphQL request for UniProt mappings did not return a valid "
            f"result! \n[Status Code] {response.status_code}"
        )
        return None
    instances = response.json()["data"]["polymer_entity_instances"] or []

    groups = {}
    for instance in instances:
        entity = instance["polymer_entity"]
        hit_seq = (entity.get("entity_poly") or {}).get(
            "pdbx_seq_one_letter_code_can"
        ) or ""
        hit_seq = "".join(hit_seq.split()).upper()
        identity, coverage = (
            _local_identity(query_seq, hit_seq) if hit_seq else (0.0, 0.0)
        )
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        for accession in identifiers.get("uniprot_ids") or [None]:
            group = groups.setdefault(
                accession,
                {
                    "uniprot_id": accession,
                    "n_chains": 0,
                    "best_identity": 0.0,
                    "best_coverage": 0.0,
                    "chains": [],
                },
            )
            group["n_chains"] += 1
            group["chains"].append(instance["rcsb_id"])
            if (identity, coverage) > (group["best_identity"], group["best_coverage"]):
                group["best_identity"], group["best_coverage"] = identity, coverage

    return sorted(
        groups.values(),
        key=lambda g: (
            g["uniprot_id"] is None,
            -g["n_chains"],
            -g["best_identity"],
            g["uniprot_id"] or "",
        ),
    )


def query_drugbank_info(lig_id, proxies=None):
    """Query the RCSB database for Drugbank info on a given ligand ID"""
    if len(lig_id) != 3:
        raise ValueError("Ligand ID has to be a three-letter code")

    query_url = f"https://data.rcsb.org/rest/v1/core/drugbank/{lig_id}"
    response = requests.get(query_url, timeout=500, proxies=proxies)
    if (code := response.status_code) != 200:
        warnings.warn(
            f'GET request on RCSB for "{lig_id}" for Drugbank Info '
            f"did not return valid result! \n[Status Code] {code}"
        )
        return
    return response.json()


def organize_drugbank_info(info_dict):
    """Organize the Drugbank info dictionary into a more readable format
    Args: info_dict (dict): The dictionary returned by the RCSB REST API
    Returns: drugbank_id, info, drug_products, targets"""
    if info_dict is None:
        warnings.warn("No Drugbank Info provided!")
        return None, None, None, None
    drugbank_id = info_dict["drugbank_container_identifiers"].get("drugbank_id")
    info = info_dict["drugbank_info"]
    targets = info_dict.get("drugbank_target")
    drug_products = None
    if "drug_products" in info:
        drug_products = info.pop("drug_products")
        drug_products = pd.DataFrame(drug_products)
    if targets:
        targets = pd.DataFrame(targets)
    return drugbank_id, info, drug_products, targets


def get_rcsb_web_data(pdb_id, proxies=None):
    """Get the data from the RCSB REST API for a given PDB ID
    Args: pdb_id (str): The PDB ID to query
    Returns: The dictionary from the RCSB REST API"""
    query_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(query_url, timeout=500, proxies=proxies)
    if (code := response.status_code) != 200:
        warnings.warn(
            f'GET request on RCSB for "{pdb_id}" for binding affinity data '
            f"did not return valid result! \n[Status Code] {code}"
        )
        return
    return response.json()


def query_binding_affinity_info(pdb_id, proxies=None):
    """Query the RCSB PDB database for binding affinity info on a given PDB ID
    Args: pdb_id (str): The PDB ID to query
    Returns: The binding affinity info as a pandas DataFrame"""
    data = get_rcsb_web_data(pdb_id, proxies=proxies)
    keyword = "rcsb_binding_affinity"
    if data is None:
        raise ValueError(
            f"Failed to fetch data for {pdb_id}. Check the PDB ID or network status."
        )
    if keyword not in data:
        return
    return pd.DataFrame(data[keyword])
