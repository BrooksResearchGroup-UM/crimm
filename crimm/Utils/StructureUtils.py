import warnings
from copy import copy
from string import ascii_uppercase, digits

import numpy as np
from Bio.Data.PDBData import (
    nucleic_letters_3to1_extended,
    protein_letters_1to3,
    protein_letters_3to1_extended,
)

from crimm.Data.components_dict import nucleic_letters_1to3
from crimm.StructEntities.Chain import PolymerChain

_BASE36_ALPHABET = digits + ascii_uppercase


def index_to_letters(index, letter=""):
    """Enumerate a sequence of letters based on the alphabet from a integer number.
    The length of the sequence corresponds to the exponent of 26 from
    the given index, and the last letter is determined by (index % 26).
    e.g.
        index_to_letters(0) = 'A'
        index_to_letters(25) = 'Z'
        index_to_letters(26) = 'AA'
        index_to_letters(701) = 'ZZ'
        index_to_letters(702) = 'AAA'
    """
    if index < 26:
        letter = ascii_uppercase[index] + letter
        return letter

    letter = ascii_uppercase[index % 26] + letter
    index = index // 26 - 1
    return index_to_letters(index, letter)


def letters_to_index(letters):
    """For a given string enumerated from alphabet, return the associated index.
    This is the inverse function of `index_to_letters`
    e.g.
        letters_to_index('A') = 0
        letters_to_index('Z') = 25
        letters_to_index('AA') = 26
        letters_to_index('ZZ') = 701
        letters_to_index('AAA') = 702
    """
    index = -1
    for i, letter in enumerate(reversed(letters)):
        cur_index = ascii_uppercase.find(letter) + 1
        index += cur_index * (26**i)
    return index


def _to_base36(value: int) -> str:
    """Encode a non-negative integer as an uppercase base36 string."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0:
        return "0"

    encoded = []
    while value:
        value, remainder = divmod(value, 36)
        encoded.append(_BASE36_ALPHABET[remainder])
    return "".join(reversed(encoded))


def compact_charmm_chain_id(chain_id: str, max_len: int = 3) -> str:
    """Return a compact uppercase token for CHARMM's 4-character SEGID limit."""
    chain_id = str(chain_id).upper()
    if len(chain_id) <= max_len and chain_id.isalnum():
        return chain_id
    if not chain_id.isalpha():
        raise ValueError(
            f"Cannot compact chain ID '{chain_id}' into a CHARMM SEGID token."
        )

    compact = _to_base36(letters_to_index(chain_id))
    if len(compact) > max_len:
        raise ValueError(
            f"Chain ID '{chain_id}' cannot be represented within CHARMM's "
            "4-character SEGID limit."
        )
    return compact


def polymer_chain_id_to_charmm_segid(chain_type: str, chain_id: str) -> str:
    """Map a polymer chain ID to a single CHARMM-safe 4-character SEGID policy.

    Rules:
        Protein A   -> PROA
        Protein AA  -> PRAA
        RNA A       -> RNAA
        RNA AA      -> RRAA
        DNA A       -> DNAA
        DNA AA      -> DRAA
        Longer IDs  -> type letter + compact token
    """
    chain_id = str(chain_id).upper()
    normalized_type = (chain_type or "").lower()

    if "polydeoxyribonucleotide" in normalized_type:
        full_prefix = "DNA"
        rollover_prefix = "DR"
        fallback_prefix = "D"
    elif "polyribonucleotide" in normalized_type:
        full_prefix = "RNA"
        rollover_prefix = "RR"
        fallback_prefix = "R"
    else:
        full_prefix = "PRO"
        rollover_prefix = "PR"
        fallback_prefix = "P"

    if len(chain_id) == 1 and chain_id.isalnum():
        return f"{full_prefix}{chain_id}"
    if len(chain_id) == 2 and chain_id.isalnum():
        return f"{rollover_prefix}{chain_id}"
    return f"{fallback_prefix}{compact_charmm_chain_id(chain_id)}"


def polymer_chain_to_charmm_segid(chain) -> str:
    """Return the CHARMM SEGID for a polymer chain object."""
    return polymer_chain_id_to_charmm_segid(chain.chain_type, chain.id)


def get_coords(entity, include_alt=False):
    """Get atom coordinates from any structure entity level or a list of entities.
    If a list of entity is provided, the coordinates will be an Nx3 array, where
    N is the total number of atoms in the list"""
    if isinstance(entity, list):
        return np.concatenate([get_coords(child) for child in entity])
    if not hasattr(entity, "level"):
        raise TypeError(
            "get_coord takes a Biopython/crimm structure object, "
            f"while {type(entity)} is provided"
        )
    if entity.level == "A":
        return entity.coord
    return np.array([a.coord for a in entity.get_atoms(include_alt)])


def _rename_chains_by_order(chains):
    for i, chain in enumerate(chains):
        chain.id = index_to_letters(i)


def rename_chains_by_order(entity):
    """Rename the chain IDs based on the order listed in the parent model so that
    the IDs start from 'A' and are continuous in the alphabetical order."""
    if entity.level not in ("S", "M"):
        raise ValueError(
            "Only Structure or Model level entities accepted! "
            f"While {entity.level} level entity {entity} is supplied. "
        )

    if entity.level == "S":
        for model in entity:
            _rename_chains_by_order(model.child_list)

    else:
        _rename_chains_by_order(entity.child_list)


def combine_hetero_chains(
    chains, new_chain_id, enforce_same_type=True, reset_resid=True
):
    """Combine a list of chains into a single chain with a specified chain ID."""
    parent = chains[0].parent
    # get an empty chain but retain the chain type of the first chain
    new_chain = copy(chains[0])
    new_chain.child_list = []
    new_chain.child_dict = {}
    new_chain.detach_parent()
    new_chain.id = new_chain_id
    all_residues = []
    all_description = []
    for chain in chains:
        if chain.chain_type in (
            "Polypeptide(L)",
            "Polyribonuleotide",
            "Polydeoxyribonucleotide",
        ):
            raise ValueError(
                f"Chain {chain.id} is a biopolymer chain and cannot be combined with "
                "other chains using this function."
            )
        if enforce_same_type and chain.chain_type != new_chain.chain_type:
            raise ValueError(
                f"Chain {chain.id} has a different chain type {chain.chain_type} "
                f"from the first chain {new_chain.id} with chain type {new_chain.chain_type}."
            )
        parent.detach_child(chain.id)
        all_residues.extend(chain.residues)

    for i, residue in enumerate(all_residues, start=1):
        residue.detach_parent()
        if reset_resid:
            het_flag, resseq, icode = residue.id
            residue.id = (het_flag, i, icode)
        if (
            hasattr(residue, "pdbx_description")
            and residue.pdbx_description not in all_description
        ):
            all_description.append(residue.pdbx_description)
        new_chain.add(residue)
    new_chain.pdbx_description = ", ".join(all_description)

    return new_chain


def get_charges(chain):
    """Get the total charge of a generated chain."""
    total_charge = 0
    for atom in chain.get_atoms():
        total_charge += atom.topo_definition.charge
    return round(total_charge, 3)


def chain_to_polymerchain(chain, reset_resid=True):
    """Convert a chain to a PolymerChain object. The chain type is determined based on
    the resnames of the residues in the chain. It will be assumed that no missing residues
    are present in the chain, and the canonical sequence will be generated based on the resnames.
    """
    resnames = set(res.resname for res in chain.residues)
    if resnames.issubset(protein_letters_3to1_extended):
        seq_dict = protein_letters_3to1_extended
        chain_type = "Polypeptide(L)"
    ## TODO(#63): Need to distinguish between DNA and RNA. It's all RNA for now.
    elif resnames.issubset(nucleic_letters_3to1_extended):
        seq_dict = nucleic_letters_3to1_extended
        chain_type = "Polyribonucleotide"
    elif resnames.issubset(nucleic_letters_1to3.values()):
        seq_dict = {k: k for k in nucleic_letters_1to3}
        chain_type = "Polyribonucleotide"
    else:
        all_names = (
            set(protein_letters_3to1_extended)
            | set(nucleic_letters_3to1_extended)
            | set(nucleic_letters_1to3.values())
        )
        unknown_resnames = resnames - all_names
        raise ValueError(
            f"Chain {chain.id} has residues with unknown resnames {unknown_resnames}. "
            "Cannot determine the chain type."
        )
    sequence = "".join(seq_dict[res.resname] for res in chain.residues)
    if not reset_resid:
        first_seq_num = chain.residues[0].id[1]
        # Fill in the missing residues at the beginning of the chain
        # with 'X' to match the sequence numbering. This is to ensure superimposition
        # is correct, which is based on the sequence alignment
        starting_seq = "X" * (first_seq_num - 1)
        sequence = starting_seq + sequence

    reported_res = [(res.id[1], res.resname) for res in chain.residues]
    polymer_chain = PolymerChain(
        chain.id,
        entity_id=1,
        author_chain_id=chain.id,
        chain_type=chain_type,
        known_sequence=sequence,
        canon_sequence=sequence,
        reported_missing_res=[],
        reported_res=reported_res,
    )
    for i, residue in enumerate(chain.residues, start=1):
        residue.detach_parent()
        if reset_resid:
            het_flag, resseq, icode = residue.id
            resseq = i
            residue.id = (het_flag, resseq, icode)
        polymer_chain.add(residue)
    return polymer_chain


_POLYMER_CHAIN_TYPES = (
    "Polypeptide(L)",
    "Polyribonucleotide",
    "Polydeoxyribonucleotide",
)
# Atoms kept when a residue is substituted; the topology generator rebuilds the rest.
_PEPTIDE_BACKBONE = ("N", "CA", "C", "O", "OXT")


def _one_letter(resname, chain_type):
    """One-letter code of a residue name; unknown names give 'X'."""
    if chain_type == "Polypeptide(L)":
        if resname in ("HSD", "HSE", "HSP"):
            return "H"
        return protein_letters_3to1_extended.get(resname, "X")
    lookup = {
        **{v: v for v in nucleic_letters_3to1_extended.values()},
        **nucleic_letters_3to1_extended,
    }
    return lookup.get(resname, "X")


def _letters_match(observed, expected):
    return observed == expected or "X" in (observed, expected)


def merge_chains(chains, new_chain_id=None, sequence=None, **renumber_kwargs):
    """Merge polymer chains into one new PolymerChain, in the order given.

    Use this when separate chains in a file are pieces of one continuous
    polymer, e.g. the NS2B and NS3 chains of a single-chain NS2B-NS3 protease
    construct. The input chains are not modified; residues are copied.

    Residues are numbered 1, 2, 3, ... in the merged chain, so overlapping
    numbering in the input chains does not collide. Each residue records where
    it came from in ``residue.xtra["source_id"]`` as
    ``(author chain id, original residue id)``.

    Parameters
    ----------
    chains : list of PolymerChain
        Chains of the same polymer type, in sequence order.
    new_chain_id : str, optional
        ID of the merged chain. Default: ID of the first chain.
    sequence : str, optional
        Canonical sequence of the full polymer. If given, the merged chain is
        passed to :func:`renumber_by_sequence` and the result is returned, so
        residue numbers follow the sequence and missing residues become gaps.
        Without it, the merged chain knows only the residues present.
    **renumber_kwargs
        Passed to :func:`renumber_by_sequence` when ``sequence`` is given.

    Returns
    -------
    PolymerChain
    """
    if len(chains) == 0:
        raise ValueError("No chains to merge.")
    chain_types = {getattr(chain, "chain_type", None) for chain in chains}
    if len(chain_types) != 1 or not chain_types <= set(_POLYMER_CHAIN_TYPES):
        raise ValueError(
            f"Only polymer chains of one type can be merged; got {sorted(map(str, chain_types))}."
        )
    chain_type = chain_types.pop()
    first = chains[0]
    if new_chain_id is None:
        new_chain_id = first.id

    residues = []
    for chain in chains:
        source = getattr(chain, "author_chain_id", None) or chain.id
        for res in sorted(chain, key=lambda r: r.id[1:]):
            new_res = res.copy()
            new_res.xtra.setdefault("source_id", (source, res.id))
            residues.append(new_res)

    seq = "".join(_one_letter(res.resname, chain_type) for res in residues)
    merged = PolymerChain(
        new_chain_id,
        entity_id=getattr(first, "entity_id", None) or 1,
        author_chain_id=new_chain_id,
        chain_type=chain_type,
        known_sequence=seq,
        canon_sequence=seq,
        reported_res=[(i, res.resname) for i, res in enumerate(residues, start=1)],
        reported_missing_res=[],
    )
    for i, res in enumerate(residues, start=1):
        res.id = (res.id[0], i, " ")
        merged.add(res)

    if sequence is not None:
        return renumber_by_sequence(merged, sequence, **renumber_kwargs)
    return merged


def _split_at_breaks(residues, bond_cutoff):
    """Split residues (in order) wherever the C(i)-N(i+1) bond is broken."""
    pieces = []
    for res in residues:
        if pieces:
            prev = pieces[-1][-1]
            if (
                "C" in prev
                and "N" in res
                and np.linalg.norm(prev["C"].coord - res["N"].coord) <= bond_cutoff
            ):
                pieces[-1].append(res)
                continue
        pieces.append([res])
    return pieces


def _source_resseq(res):
    source_chain, (_het, resseq, _icode) = res.xtra.get("source_id", (None, res.id))
    return source_chain, resseq


def _place_pieces(pieces, piece_seqs, sequence, min_identity):
    """Place each piece on the sequence without gaps, keeping the pieces in order.

    Returns one (start, n_identical) per piece, start 0-based. The best
    placement maximises the identical residues; ties are broken by how many
    consecutive pieces from the same input chain keep their original spacing.
    Raises ValueError if a piece cannot be placed or the best choice is not unique.
    """
    candidates = []
    for k, seq in enumerate(piece_seqs):
        n = len(seq)
        found, best_seen = [], 0
        for start in range(len(sequence) - n + 1):
            same = sum(
                _letters_match(a, b) for a, b in zip(seq, sequence[start : start + n])
            )
            best_seen = max(best_seen, same)
            if same >= min_identity * n:
                found.append((start, same))
        if not found:
            src = _source_resseq(pieces[k][0])
            raise ValueError(
                f"Piece {k} ({n} residues from chain {src[0]}, starting at residue "
                f"{src[1]}: {seq}) matches no stretch of the sequence with identity "
                f">= {min_identity} (best: {best_seen}/{n})."
            )
        candidates.append(found)

    def spacing_bonus(k, prev_start, start):
        src_prev, last_prev = _source_resseq(pieces[k - 1][-1])
        src_cur, first_cur = _source_resseq(pieces[k][0])
        if src_prev != src_cur:
            return 0
        seq_gap = start - (prev_start + len(piece_seqs[k - 1]) - 1)
        return int(seq_gap == first_cur - last_prev)

    # states: start of the current piece -> (score, n_paths, path)
    states = {start: ((same, 0), 1, [(start, same)]) for start, same in candidates[0]}
    for k in range(1, len(pieces)):
        new_states = {}
        for start, same in candidates[k]:
            best, n_paths = None, 0
            for prev_start, (score, count, path) in states.items():
                if prev_start + len(piece_seqs[k - 1]) > start:
                    continue
                total = (
                    score[0] + same,
                    score[1] + spacing_bonus(k, prev_start, start),
                )
                if best is None or total > best[0]:
                    best, n_paths = (total, path + [(start, same)]), count
                elif total == best[0]:
                    n_paths += count
            if best is not None:
                new_states[start] = (best[0], n_paths, best[1])
        if not new_states:
            raise ValueError(
                f"Piece {k} cannot be placed after piece {k - 1}; are the chains "
                "given in sequence order?"
            )
        states = new_states

    top = max(score for score, _, _ in states.values())
    winners = [(count, path) for score, count, path in states.values() if score == top]
    if sum(count for count, _ in winners) > 1:
        ambiguous = [
            f"piece {k} ({piece_seqs[k]}): starts {[s + 1 for s, _ in c]}"
            for k, c in enumerate(candidates)
            if len(c) > 1
        ]
        raise ValueError(
            "The placement of the pieces on the sequence is not unique. Candidates: "
            + "; ".join(ambiguous)
            + ". Remove or trim the short pieces, or raise min_identity."
        )
    return winners[0][1]


def renumber_by_sequence(
    chain,
    sequence,
    min_identity=0.8,
    substitute_mismatches=False,
    bond_cutoff=1.8,
    new_chain_id=None,
):
    """Number a protein chain by a canonical sequence, so missing residues become gaps.

    For structures without sequence information (PDB files) or with numbering
    that does not follow the sequence. The chain is cut into unbroken pieces
    wherever the C-N peptide bond is longer than ``bond_cutoff``; each piece is
    placed on the sequence without gaps, in order, at the position with the most
    identical residues. No alignment parameters need tuning. The input chain is
    not modified.

    Residues that differ from the sequence (mutations, engineered residues) are
    kept by default and the chain's canonical sequence records the residue that
    is present, so they are never treated as missing. With
    ``substitute_mismatches=True`` they are renamed to the sequence's residue and
    reduced to backbone atoms (plus CB unless the new residue is GLY); the
    topology generator rebuilds the side chain. A warning lists them either way.

    Parameters
    ----------
    chain : PolymerChain
        Polypeptide chain to renumber, e.g. from :func:`merge_chains`.
    sequence : str
        Canonical one-letter sequence of the whole chain. ``X`` matches anything.
    min_identity : float
        Lowest fraction of identical residues for a piece to be placed.
    substitute_mismatches : bool
        Replace residues that differ from the sequence (see above).
    bond_cutoff : float
        Largest C-N distance, in Å, still treated as a peptide bond.
    new_chain_id : str, optional
        ID of the new chain. Default: the input chain's ID.

    Returns
    -------
    PolymerChain
        New chain numbered 1..len(sequence). ``chain.sequence_mapping`` lists,
        for each piece, where it was placed, its identity and its mismatches.
    """
    chain_type = getattr(chain, "chain_type", None)
    if chain_type != "Polypeptide(L)":
        raise NotImplementedError(
            f"renumber_by_sequence supports polypeptide chains only, not {chain_type}."
        )
    sequence = str(sequence).strip().upper()
    residues = sorted(chain, key=lambda r: r.id[1:])
    if len(residues) == 0:
        raise ValueError(f"{chain} has no residues.")
    source = getattr(chain, "author_chain_id", None) or chain.id
    copies = []
    for res in residues:
        new_res = res.copy()
        new_res.xtra.setdefault("source_id", (source, res.id))
        copies.append(new_res)

    pieces = _split_at_breaks(copies, bond_cutoff)
    piece_seqs = [
        "".join(_one_letter(r.resname, chain_type) for r in p) for p in pieces
    ]
    placements = _place_pieces(pieces, piece_seqs, sequence, min_identity)

    def three_letter(letter):
        return protein_letters_1to3.get(letter, "UNK").upper()

    canon = list(sequence)
    resnames = [three_letter(letter) for letter in sequence]
    mapping, mismatches, offsets = [], [], {}
    for k, (piece, seq, (start, same)) in enumerate(
        zip(pieces, piece_seqs, placements)
    ):
        piece_mismatches = []
        for i, (res, letter) in enumerate(zip(piece, seq)):
            pos = start + i + 1
            src_chain, src_resseq = _source_resseq(res)
            offsets.setdefault(src_chain, set()).add(pos - src_resseq)
            res.id = (res.id[0], pos, " ")
            if _letters_match(letter, sequence[pos - 1]):
                continue
            expected = resnames[pos - 1]
            piece_mismatches.append((pos, res.resname, expected))
            if substitute_mismatches:
                _substitute_residue(res, expected)
            else:
                canon[pos - 1] = letter
                resnames[pos - 1] = res.resname
        mismatches.extend(piece_mismatches)
        mapping.append(
            {
                "piece": k,
                "source_first": _res_source_label(piece[0]),
                "source_last": _res_source_label(piece[-1]),
                "start": start + 1,
                "end": start + len(piece),
                "identity": round(same / len(piece), 3),
                "mismatches": piece_mismatches,
            }
        )

    if mismatches:
        listed = ", ".join(
            f"{obs}{pos} (sequence: {exp})" for pos, obs, exp in mismatches
        )
        action = (
            "replaced by the sequence's residue; their side chains are rebuilt by the "
            "topology generator"
            if substitute_mismatches
            else "kept as in the structure (substitute_mismatches=True replaces them)"
        )
        warnings.warn(f"Residues that differ from the sequence, {action}: {listed}.")
    for src_chain, chain_offsets in offsets.items():
        if len(chain_offsets) > 1:
            warnings.warn(
                f"Pieces of chain {src_chain} were placed with different numbering "
                f"offsets {sorted(chain_offsets)}. Check chain.sequence_mapping."
            )

    canon_seq = "".join(canon)
    new_chain = PolymerChain(
        new_chain_id or chain.id,
        entity_id=getattr(chain, "entity_id", None) or 1,
        author_chain_id=getattr(chain, "author_chain_id", None) or chain.id,
        chain_type=chain_type,
        known_sequence=canon_seq,
        canon_sequence=canon_seq,
        reported_res=list(enumerate(resnames, start=1)),
        reported_missing_res=[],
    )
    for piece in pieces:
        for res in piece:
            new_chain.add(res)
    new_chain.sort_residues()
    new_chain.sequence_mapping = mapping
    return new_chain


def _res_source_label(res):
    """'chain:resseq' of the residue in its original chain."""
    src_chain, resseq = _source_resseq(res)
    return f"{src_chain}:{resseq}"


def _substitute_residue(res, new_resname):
    """Rename a residue and keep only backbone atoms (and CB unless GLY)."""
    keep = set(_PEPTIDE_BACKBONE)
    if new_resname != "GLY":
        keep.add("CB")
    for atom in list(res.get_atoms()):
        if atom.get_id() not in keep:
            res.detach_child(atom.get_id())
    res.resname = new_resname
    het, resseq, icode = res.id
    if het.startswith("H_"):
        res.id = (" ", resseq, icode)
