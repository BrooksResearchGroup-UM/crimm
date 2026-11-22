import json
import warnings

import numpy as np
import requests
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq

from crimm.Fetchers import fetch_alphafold, fetch_rcsb, uniprot_id_query
from crimm.Modeller.TopoLoader import ResidueTopologySet
from crimm.StructEntities.Chain import PolymerChain
from crimm.Superimpose.ChainSuperimposer import ChainSuperimposer


def find_gaps_within_range(gaps, segment):
    in_range = []
    for gap in gaps:
        if gap.issubset(segment):
            in_range.append(gap)
    return in_range


def find_gaps_range_overlap(gaps, segment):
    overlaps = []
    for gap in gaps:
        if segment.intersection(gap):
            overlaps.append(gap)
    return overlaps


def find_segment_offsets(chainA_seq, chainB_seq):
    aligner = PairwiseAligner()
    aligner.target_internal_open_gap_score = -100
    aligner.query_internal_open_gap_score = -100
    align = aligner.align(chainA_seq, chainB_seq)[0]
    chainA_aligned, chainB_aligned = align.aligned + 1  # resseq is 1-indexed
    offsets = (chainB_aligned - chainA_aligned)[:, 0]
    offset_dict = {
        range(*segmentA): offset for segmentA, offset in zip(chainA_aligned, offsets)
    }
    return offset_dict


def find_gap_offsets(gaps, segment_offset_dict):
    gap_offset_dict = {}
    for segment, offset in segment_offset_dict.items():
        segment = set(segment)
        gaps_in_range = find_gaps_within_range(gaps, segment)
        gap_offset_dict.update({tuple(gap): offset for gap in gaps_in_range})
        for gap in gaps_in_range:
            gaps.remove(gap)
        if len(gaps) == 0:
            break
    return gap_offset_dict


def translate_gap_ids(chainA, chainB):
    """Translate gap ids from chainA to chainB based on their canonical sequences
    alignment"""
    if chainA.can_seq == chainB.can_seq:
        rp_gap = translate_gaps_with_identical_seqs(chainA, chainB)
        translated = {tuple(sorted(gap)): tuple(sorted(gap)) for gap in rp_gap}
        return translated

    segment_offsets = find_segment_offsets(chainA.can_seq, chainB.can_seq)
    gap_offsets = find_gap_offsets(chainA.gaps, segment_offsets)
    translated = {}
    for gap_A, offset in gap_offsets.items():
        translated_gap_A = {resseq + offset for resseq in gap_A}
        if find_gaps_range_overlap(chainB.gaps, translated_gap_A):
            continue
        translated[tuple(sorted(gap_A))] = tuple(sorted(translated_gap_A))
    return translated


def translate_gaps_with_identical_seqs(chainA, chainB):
    model_chain_gaps = chainA.gaps
    template_chain_gaps = chainB.gaps
    translated = []
    i, j = 0, 0

    while i < len(model_chain_gaps) and j < len(template_chain_gaps):
        # gap of the model chain
        m_gap = model_chain_gaps[i]
        # start idx of the gap
        m_start = min(m_gap)
        # gap of the template chain
        t_gap = template_chain_gaps[j]
        # end idx of the gap
        t_end = max(t_gap)

        ################# Possible Scenarios for t_gap and m_gap ###################
        #   ---------------[cur t_gap]------[next t_gap]-------  t_gap positions
        # 1 ---------[cur m_gap]-------------------------------  overlap
        # 2 --------------------[cur m_gap]--------------------  overlap
        # 3 ----------------------------[cur m_gap]------------  m_start > p_end
        # 4 ---[cur m_gap]---------[next m_gap]----------------  repair cur m_gap
        ############################################################################

        # If gaps overlap, move on to the next gap for both (e.g 1 and 2)
        if m_gap.intersection(t_gap):
            i += 1
            j += 1
        # If the p_gap ends before the current model_gap,
        # move on to the next t_gap (e.g 3)
        elif m_start > t_end:
            j += 1
        # Otherwise, current model_gap can be fixed by the template_chain,
        # and move on to the next m_gap (e.g. 4)
        else:
            translated.append(m_gap)
            i += 1

    # If there are left-over gaps from receiving chain after going over
    # the last gap of providing chain, these gaps can all be repaired.
    if i < len(model_chain_gaps):
        translated.extend(model_chain_gaps[i:])
    return translated


def report_loops(chain: PolymerChain):
    res_dict = dict(chain.missing_res)
    loops = []
    for gap in chain.gaps:
        res_ids = sorted(gap)
        missing_res = [res_dict[res_id] for res_id in res_ids]
        anchor, target = chain[min(gap) - 1], chain[max(gap) + 1]
        loops.append(
            {
                "anchor": {"C": anchor["C"].coord, "CA": anchor["CA"].coord},
                "loop_res_ids": res_ids,
                "loop": missing_res,
                "target": {"N": target["N"].coord, "CA": target["CA"].coord},
            }
        )
    return loops


def _clean_sequence(sequence):
    """One-letter sequence without whitespace, upper case; rejects other characters."""
    seq = "".join(str(sequence).split()).upper()
    if not seq or not seq.isalpha():
        raise ValueError(f"Not a one-letter sequence: {sequence!r}")
    return seq


def _parse_pdbids(pdbids):
    """{PDBID: [chain ids] or None} from "1ABC", "1ABC.A", "1ABC_A", a list or a dict."""
    if isinstance(pdbids, dict):
        return {
            str(k).upper(): (
                None if v is None else [v] if isinstance(v, str) else list(v)
            )
            for k, v in pdbids.items()
        }
    if isinstance(pdbids, str):
        pdbids = [pdbids]
    entries = {}
    for item in pdbids:
        item = str(item).strip()
        pdbid, _, chain_id = item.replace("_", ".").partition(".")
        pdbid = pdbid.upper()
        if len(pdbid) != 4:
            raise ValueError(f"Not a PDB ID: {item!r}")
        if not chain_id:
            entries[pdbid] = None
        elif entries.get(pdbid, []) is not None:
            entries.setdefault(pdbid, []).append(chain_id)
    return entries


def _load_template_chain(template):
    """Return (first polymer chain, structure id) from a file path or crimm entity."""
    import os

    from crimm.StructEntities.Chain import BaseChain

    if isinstance(template, (str, os.PathLike)):
        path = os.fspath(template)
        if not path.lower().endswith((".cif", ".mmcif")):
            raise ValueError(
                f"{path}: only mmCIF templates can be read directly, because the "
                "template needs a sequence. For a PDB file, load it, run "
                "crimm.Utils.StructureUtils.renumber_by_sequence and pass the chain."
            )
        from crimm.IO.MMCIFParser import MMCIFParser

        structure = MMCIFParser(QUIET=True).get_structure(path)
        entity, structure_id = structure, os.path.basename(path)
    else:
        entity = template
        top = template.get_top_parent() if hasattr(template, "get_top_parent") else None
        structure_id = getattr(top, "id", None) or getattr(template, "id", "template")
    if isinstance(entity, BaseChain):
        chains = [entity]
    else:
        level = getattr(entity, "level", None)
        if level == "S":
            entity = entity.child_list[0]
        chains = list(entity)
    for chain in chains:
        if isinstance(chain, PolymerChain) and chain.can_seq:
            return chain, structure_id
    raise ValueError(f"No polymer chain with a sequence found in {template}.")


class ChainLoopBuilder:
    """
    loop modeller for PDB protein structures by homology modeling
    """

    # C-N distance (Å) above which a copied gap is reported as not joined.
    junction_warn = 2.0

    def __init__(
        self,
        model_chain: PolymerChain,
        model_can_seq=None,
        pdbid=None,
        inplace=False,
        uniprot_id=None,
    ):
        """
        Args:
            model_chain: the chain to be repaired
            model_can_seq (optional): the canonical sequence of the chain.
                If not provided, it will be assumed to be the canonical sequence
                parsed from mmcif file.
            pdbid (optional): the PDB ID of the chain. If not provided, it will
                look up the PDB ID from the chain's top parent
                (model or structure containing the chain).
            inplace: if True, the model_chain will be modified in place. Default
                is False.
            uniprot_id (optional): UniProt accession of the chain, used by
                build_from_alphafold() instead of looking it up from the PDB ID.
                Give this for chains that do not come from the PDB.
        """
        if not inplace:
            self.model_chain = model_chain.copy()
        else:
            self.model_chain = model_chain
        if model_can_seq is not None:
            self.model_can_seq = model_can_seq
        else:
            self.model_can_seq = model_chain.can_seq
        self.repaired_gaps = []
        self.repaired_residues = {}
        self.imposer = ChainSuperimposer()
        self.template_chain = None
        self.template_can_seq = None
        self.translated_ids = None
        self.query_results = None
        self.uniprot_id = uniprot_id
        self.build_report = []
        self.pdbid = pdbid
        if self.pdbid is None:
            if (parent := (model_chain.get_top_parent())).level == "S":
                self.pdbid = parent.id
            elif (
                parent := model_chain.get_top_parent()
            ).level == "M" and parent.pdb_id is not None:
                self.pdbid = parent.pdb_id
            elif uniprot_id is None:
                warnings.warn(
                    f"PDB ID not set for {model_chain}! "
                    "Make sure the chain is loaded from a MMCIF file or from fetch() "
                    "to have the PDB ID correctly parsed, or set the PDB ID manually."
                )

        if self.model_can_seq is None:
            raise AttributeError("Canonical sequence is required to repair loops")

    def set_template_chain(self, template_chain: PolymerChain):
        """
        Set the template chain for copying the coordinates of loop residues from
        """
        self.template_chain = template_chain
        self.template_can_seq = template_chain.can_seq
        if self.template_can_seq is None:
            raise AttributeError(
                f"Canonical sequence does not exist for {template_chain}! "
                "Make sure the chain is loaded from a MMCIF file or from fetch() "
                "to have the canonical sequence correctly parsed."
            )

        self.translated_ids = translate_gap_ids(self.model_chain, self.template_chain)

    def superimpose_two_chains(self, on_atoms="backbone"):
        """
        Impose template chain onto modeling chain
        Return the RMSD value
        """

        if not hasattr(self, "template_chain"):
            raise AttributeError(
                "Template chain not set! Use set_template_chain() to load template first"
            )

        self.imposer.set_chains(
            self.model_chain, self.template_chain, on_atoms=on_atoms
        )
        self.imposer.apply_transform(self.template_chain)
        return self.imposer.rms

    def _copy_gap_residues(self, gap, residues):
        # gap is a set and has to be sorted before calculated the idx for
        # insertion. Otherwise, it would mess up the residue sequence in
        # the structure

        residues = sorted(residues, key=lambda x: x.id[1])
        gap = sorted(gap)

        for i, res in zip(gap, residues):
            hetflag = res.id[0]
            res.id = (hetflag, i, " ")
            # i in insert refer to child list index, thus use i-1 here
            self.model_chain.add(res)
        self.model_chain.sort_residues()
        self.repaired_gaps.append(tuple(gap))

    def get_repair_residues_from_template(
        self,
        rmsd_threshold=None,
    ):
        """
        Find repairable residue for model_chain from template_chain gaps.
        """
        if not hasattr(self, "template_chain"):
            raise AttributeError(
                "Template chain not set!Use set_template_chain() to load template first"
            )

        rmsd_qualified = {}
        self.unanchored_gaps = []
        for gap, template_resseqs in self.translated_ids.items():
            try:
                self.imposer.set_around_gap(
                    self.model_chain, self.template_chain, gap, cutoff=5
                )
            except ValueError:
                # no residue present in both chains to superimpose on
                self.unanchored_gaps.append(gap)
                continue
            self.imposer.apply_transform(self.template_chain)
            rmsd = self.imposer.rms
            if rmsd_threshold is not None and rmsd > rmsd_threshold:
                warnings.warn(
                    f"RMSD ({rmsd}) of superposition around gap {gap} is "
                    f"higher than the threshold {rmsd_threshold}. "
                    "Repair not applied!"
                )
                continue

            available_residues = []
            for i in template_resseqs:
                resseq = int(i)
                if resseq not in self.template_chain:
                    break
                available_residues.append(self.template_chain[resseq].copy())

            if len(available_residues) != len(template_resseqs):
                continue

            rmsd_qualified[gap] = {"residues": available_residues, "rmsd": rmsd}

        return rmsd_qualified

    def highlight_repaired_gaps(self, color="green", add_licorice=False):
        """
        Highlight the repaired gaps with color (default: cyan)
        """
        from crimm.Visualization import get_viewer

        fixed_res = []
        for res_list in self.repaired_gaps:
            for res_seq in res_list:
                fixed_res.append(self.model_chain[res_seq])
        if len(fixed_res) < 5:
            add_licorice = True  # cartoon will not render if there are too few residues
        view = get_viewer()
        view.load_entity(self.model_chain)
        view.highlight_residues(fixed_res, color=color, add_licorice=add_licorice)

        return view

    def show_gaps(self):
        """
        Highlight the residues on the gap edges with color (default: red)
        """
        if len(self.model_chain.gaps) == 0:
            warnings.warn(f"No gap found in {self.model_chain}!")
        from crimm.Visualization import get_viewer

        view = get_viewer()

        view.load_entity(self.model_chain)
        view.subdue_all_entities()
        rgb_red = [1, 0, 0]
        line_width = 0.5
        for gap in self.model_chain.gaps:
            # Center on the gap by selecting the terminal residues around the gap
            if (st := min(gap) - 1) in self.model_chain and (
                end := max(gap) + 1
            ) in self.model_chain:
                st_coord = self.model_chain[st]["CA"].coord
                end_coord = self.model_chain[end]["CA"].coord
                arrow_name = f"gap:{st + 1}-{end - 1}"
                view.add_arrow(st_coord, end_coord, rgb_red, line_width, arrow_name)

        return view

    def show(self, **kwargs):
        """Show the model and template chains with the active visualization backend.

        kwargs (``width``, ``height``) go to :func:`crimm.Visualization.show_multiple`.
        """
        from crimm.Visualization import show_multiple

        return show_multiple([self.model_chain, self.template_chain], **kwargs)

    @staticmethod
    def _build_seq_query(
        seq_string, max_num_match, identity_score_cutoff, uniprot_id=None
    ):
        """
        Args:
            seq_string: str
            max_num_match: int
            identity_score_cutoff: float (from 0 to 1)
            uniprot_id: if given, keep only polymer entities mapped to this
                UniProt accession
        """
        query = {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity_score_cutoff,
                "sequence_type": "protein",
                "value": seq_string,
            },
        }
        if uniprot_id is not None:
            accession = {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers."
                    "reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": str(uniprot_id).strip().upper(),
                },
            }
            query = {
                "type": "group",
                "logical_operator": "and",
                "nodes": [query, accession],
            }
        query_dict = {
            "query": query,
            "request_options": {
                "scoring_strategy": "sequence",
                "paginate": {"start": 0, "rows": max_num_match},
            },
            "return_type": "polymer_instance",
        }

        return query_dict

    def query_seq_match(
        self, max_num_match, identity_score_cutoff, sequence=None, uniprot_id=None
    ):
        """
        Make RCSB PDB query for protein sequence

        :param max_num_match : maximum number of top matches returned.
        :param identity_score_cutoff: lowest identity score accepted
        :param sequence: one-letter query sequence; default: the chain's
            canonical sequence
        :param uniprot_id: keep only hits mapped to this UniProt accession
        :rtype retults : dict consists (sequence_matching_score, PDB_ID, entity_ID)
        """
        if sequence is None:
            seq_string = str(self.model_can_seq)
        else:
            seq_string = _clean_sequence(sequence)
        json_q = ChainLoopBuilder._build_seq_query(
            seq_string, max_num_match, identity_score_cutoff, uniprot_id=uniprot_id
        )
        # entry point of rcsb.org for advancced search
        url = "https://search.rcsb.org/rcsbsearch/v2/query?"
        response = requests.post(url, json=json_q, timeout=500)
        if getattr(response, "status_code", 200) == 204:
            # RCSB answers "204 No Content" when nothing matches
            return {}
        r_dict = json.loads(response.text)

        if "result_set" not in r_dict:
            raise ValueError(
                "Query on the sequence did not return "
                f"the requested information. {response}"
            )

        results = {}
        for entry in r_dict["result_set"]:
            pdb_id, chain_id = entry["identifier"].split(".")
            if pdb_id == self.pdbid:
                continue
            if entry["score"] not in results:
                results[entry["score"]] = {}
            if pdb_id not in results[entry["score"]]:
                results[entry["score"]][pdb_id] = []
            results[entry["score"]][pdb_id].append(chain_id)
        return results

    @staticmethod
    def get_templates(query_results: dict, local_entry_point: str):
        """
        Return a generator of all template chains from the query result dict
        """
        for entity_dict in query_results.values():
            yield from ChainLoopBuilder._iter_pdb_templates(
                entity_dict, local_entry_point
            )

    @staticmethod
    def _iter_pdb_templates(entries: dict, local_entry_point=None):
        """Yield (pdbid, chain) for {pdbid: [chain ids] or None}; None means
        every polypeptide chain of the first model. Entries or chains that
        cannot be loaded are skipped with a warning."""
        for pdbid, chain_ids in entries.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    first_model = fetch_rcsb(
                        pdbid,
                        use_bio_assembly=False,
                        include_solvent=False,
                        local_entry=local_entry_point,
                        first_model_only=True,
                    )
            except Exception as exc:  # network, missing file, parse error
                warnings.warn(f"Template {pdbid} skipped: could not load it ({exc}).")
                continue
            if chain_ids is None:
                chain_ids = [
                    chain.id
                    for chain in first_model
                    if getattr(chain, "chain_type", None) == "Polypeptide(L)"
                ]
            for chain_id in chain_ids:
                if chain_id not in first_model:
                    warnings.warn(
                        f"Template {pdbid} has no chain {chain_id}; chain IDs are "
                        "label_asym_id, as in RCSB polymer instance IDs "
                        f"(available: {[c.id for c in first_model]})."
                    )
                    continue
                yield (pdbid, first_model[chain_id])

    def _reject_by_rmsd(self, overall_rmsd_threshold):
        if overall_rmsd_threshold is not None:
            rms = self.superimpose_two_chains()
            if rms > overall_rmsd_threshold:
                return True
        return False

    def _update_candidates(self, cur_repairables, candidates, pdbid):
        for gap_key, info_dict in cur_repairables.items():
            if gap_key not in candidates or (
                candidates[gap_key]["rmsd"] > info_dict["rmsd"]
            ):
                info_dict["structure_id"] = pdbid
                candidates[gap_key] = info_dict

    def build_from_homology(
        self,
        max_num_match=10,
        identity_score_cutoff=0.95,
        rmsd_threshold=None,
        overall_rmsd_threshold=None,
        local_entry_point=None,
        sequence=None,
        uniprot_id=None,
        include_terminal=True,
        allow_mismatch=False,
    ):
        """Build the missing loops of the model chain from homology models from
        PDB. Seqeunce search query is performed first to find the closest homology
        models based on sequence identity scores.

        Args:
            max_num_match: number of top search hits (polymer instances) to try.
            identity_score_cutoff: lowest sequence identity of a hit (0 to 1).
            rmsd_threshold: skip a gap if the superposition RMSD of the residues
                around it is above this value (Å).
            overall_rmsd_threshold: skip a template whose whole-chain
                superposition RMSD is above this value (Å).
            local_entry_point: path of a local PDB mmCIF archive; default rcsb.org.
            sequence: one-letter sequence to search with instead of the chain's
                canonical sequence, e.g. a construct or a related sequence.
            uniprot_id: keep only hits whose PDB entity is mapped to this
                UniProt accession. The PDB often maps entries to a strain- or
                isoform-specific accession (e.g. viral polyproteins), so check
                the accession in the entries you expect if nothing is found.
                The ``uniprot_id`` given to the constructor is not used here.
            include_terminal: also copy missing terminal residues.
            allow_mismatch: copy a gap even if the template's residues there
                differ from the model chain's sequence. Default: such gaps
                are taken from another template or left unbuilt.

        For each gap, the template with the lowest local RMSD is used. Per-gap
        results are added to ``self.build_report``.
        """
        # if local_entry_point is None, rcsb.org will be used
        # is_continuous() ignores missing termini, which this method can build.
        if not self.model_chain.missing_res or (
            not include_terminal and self.model_chain.is_continuous()
        ):
            warnings.warn(f"{self.model_chain} does not have any missing loops")
            return

        self.query_results = self.query_seq_match(
            max_num_match,
            identity_score_cutoff,
            sequence=sequence,
            uniprot_id=uniprot_id,
        )

        if not self.query_results:
            hint = ""
            if uniprot_id is not None:
                hint = (
                    f" No hit is mapped to UniProt {uniprot_id}. The PDB often maps "
                    "entries to a strain- or isoform-specific accession; run without "
                    "uniprot_id, or check the accession of the entries you expect."
                )
            warnings.warn(
                f"No homology model found for {self.pdbid}-{self.model_chain.id} "
                f"from sequence search!{hint}"
            )
            return

        all_templates = self.get_templates(self.query_results, local_entry_point)
        self._build_from_pdb_templates(
            all_templates,
            include_terminal=include_terminal,
            rmsd_threshold=rmsd_threshold,
            overall_rmsd_threshold=overall_rmsd_threshold,
            allow_mismatch=allow_mismatch,
        )

    def build_from_pdbids(
        self,
        pdbids,
        rmsd_threshold=None,
        overall_rmsd_threshold=None,
        local_entry_point=None,
        include_terminal=True,
        allow_mismatch=False,
    ):
        """Build the missing loops of the model chain from PDB entries you choose.

        Args:
            pdbids: PDB IDs of the templates. Accepts "1ABC" (every polypeptide
                chain of the entry is tried), "1ABC.A" or "1ABC_A" (one chain),
                a list of these, or a dict {"1ABC": ["A", "B"]}. Chain IDs are
                label_asym_id, the same as RCSB polymer instance IDs.
            rmsd_threshold, overall_rmsd_threshold, local_entry_point,
            include_terminal, allow_mismatch: as in build_from_homology().

        For each gap, the template with the lowest local RMSD is used. Per-gap
        results are added to ``self.build_report``.
        """
        if len(self.model_chain.missing_res) == 0:
            warnings.warn(f"{self.model_chain} does not have any missing residues")
            return
        entries = _parse_pdbids(pdbids)
        templates = self._iter_pdb_templates(entries, local_entry_point)
        self._build_from_pdb_templates(
            templates,
            include_terminal=include_terminal,
            rmsd_threshold=rmsd_threshold,
            overall_rmsd_threshold=overall_rmsd_threshold,
            allow_mismatch=allow_mismatch,
        )

    def _build_from_pdb_templates(
        self,
        templates,
        include_terminal,
        rmsd_threshold,
        overall_rmsd_threshold,
        allow_mismatch,
    ):
        """Try each (pdbid, chain); fill each gap from the template with the
        lowest local RMSD."""
        repair_candidates, skipped = {}, {}
        tried = []
        for pdbid, template in templates:
            tried.append(f"{pdbid}.{template.id}")
            self.set_template_chain(template)
            if self._reject_by_rmsd(overall_rmsd_threshold):
                continue
            cur_repairables, cur_skipped = self._collect_repairables(
                structure_id=pdbid,
                include_terminal=include_terminal,
                rmsd_threshold=rmsd_threshold,
                allow_mismatch=allow_mismatch,
                bfactor_label="B-factor",
                warn_mismatch=False,
            )
            for gap, mismatches in cur_skipped.items():
                skipped.setdefault(gap, []).append(
                    (f"{pdbid}.{template.id}", mismatches)
                )
            if cur_repairables:
                self._update_candidates(cur_repairables, repair_candidates, pdbid)

        unfilled = {g: v for g, v in skipped.items() if g not in repair_candidates}
        for gap, per_template in sorted(unfilled.items()):
            label, mismatches = per_template[0]
            listed = ", ".join(f"{e}{i} (template: {t})" for i, e, t in mismatches)
            warnings.warn(
                f"Gap {min(gap)}-{max(gap)} not built: the residues there differ "
                f"from the sequence in every template that covers it "
                f"({len(per_template)}; e.g. {label} at {listed}). Use "
                "allow_mismatch=True to copy them anyway."
            )
        if not repair_candidates:
            warnings.warn(
                f"No repairable residues found for {self.model_chain} from "
                f"templates {tried}."
            )
            return
        self._apply_repairables(repair_candidates)

    def build_from_alphafold(self, include_terminal=False, uniprot_id=None):
        """Build the missing loops of the model chain from AlphaFold models.
        The AlphaFold models are downloaded from the AlphaFold database. If
        include_terminal is True, the missing terminal residues will be copied
        from the AlphaFold model.

        The UniProt accession is taken from ``uniprot_id`` if given, then from
        the ``uniprot_id`` given to the constructor, and otherwise looked up
        from the PDB ID. Per-gap results are added to ``self.build_report``.
        """

        if len(self.model_chain.missing_res) == 0:
            warnings.warn(f"{self.model_chain} does not have any missing residues")
            return

        uniprot_id = uniprot_id or self.uniprot_id
        if uniprot_id is None:
            if self.pdbid is None:
                warnings.warn(
                    "Missing looper not built due to no PDB ID or UniProt ID "
                    f"provided for {self.model_chain}."
                )
                return
            uniprot_id = uniprot_id_query(self.pdbid, self.model_chain.entity_id)
        af_struct = fetch_alphafold(uniprot_id)
        if af_struct is None:
            warnings.warn(
                "Missing looper not built due to no AlphaFold structure avaible "
                f"for {self.model_chain}."
            )
            return

        af_chain = af_struct.child_list[0].child_list[0]
        self._build_from_template_chain(
            af_chain,
            structure_id=af_struct.id,
            include_terminal=include_terminal,
            rmsd_threshold=None,
            allow_mismatch=False,
            bfactor_label="pLDDT",
        )

    def build_from_template(
        self,
        template,
        include_terminal=False,
        rmsd_threshold=None,
        allow_mismatch=False,
        bfactor_label=None,
    ):
        """Build the missing loops of the model chain from a template you provide.

        Use this when the template is not in the PDB or the AlphaFold database,
        e.g. an AlphaFold Server or other predicted model of the full sequence.

        Args:
            template: an mmCIF file path, or a PolymerChain, Model or Structure
                (the first polymer chain is used). The template needs a
                canonical sequence; for a PDB file, load it and run
                crimm.Utils.StructureUtils.renumber_by_sequence first.
            include_terminal: if True, also copy missing terminal residues.
            rmsd_threshold: skip a gap if the superposition RMSD of the residues
                around it is above this value (Å). Default: no limit.
            allow_mismatch: copy a gap even if the template's residue names
                there differ from the model chain's sequence. Default: such
                gaps are skipped with a warning.
            bfactor_label: name of the per-residue value in the template's
                B-factor column, reported per gap (e.g. "pLDDT"). Default:
                "pLDDT" for files whose name starts with "fold_" or "AF-",
                otherwise "B-factor".

        Per-gap results are added to ``self.build_report``; ``report()`` prints them.
        """
        if len(self.model_chain.missing_res) == 0:
            warnings.warn(f"{self.model_chain} does not have any missing residues")
            return
        template_chain, structure_id = _load_template_chain(template)
        if bfactor_label is None:
            name = str(structure_id).rsplit("/", 1)[-1]
            bfactor_label = "pLDDT" if name.startswith(("fold_", "AF-")) else "B-factor"
        self._build_from_template_chain(
            template_chain,
            structure_id=structure_id,
            include_terminal=include_terminal,
            rmsd_threshold=rmsd_threshold,
            allow_mismatch=allow_mismatch,
            bfactor_label=bfactor_label,
        )

    def _gap_name_mismatches(self, gap, residues):
        """(resseq, model resname, template resname) where the names differ."""
        expected = dict(self.model_chain.missing_res)
        out = []
        for resseq, res in zip(sorted(gap), sorted(residues, key=lambda r: r.id[1])):
            exp = expected.get(resseq)
            name = "HIS" if res.resname in ("HSD", "HSE", "HSP") else res.resname
            if exp is not None and name != exp:
                out.append((resseq, exp, res.resname))
        return out

    def _build_from_template_chain(
        self,
        template_chain,
        structure_id,
        include_terminal,
        rmsd_threshold,
        allow_mismatch,
        bfactor_label,
    ):
        self.set_template_chain(template_chain)
        repairables, _ = self._collect_repairables(
            structure_id=structure_id,
            include_terminal=include_terminal,
            rmsd_threshold=rmsd_threshold,
            allow_mismatch=allow_mismatch,
            bfactor_label=bfactor_label,
            warn_mismatch=True,
        )
        if len(repairables) == 0:
            warnings.warn(
                f"No repairable residues found for {self.model_chain} "
                f"from template {structure_id}."
            )
            return
        self._apply_repairables(repairables)

    def _collect_repairables(
        self,
        structure_id,
        include_terminal,
        rmsd_threshold,
        allow_mismatch,
        bfactor_label,
        warn_mismatch,
    ):
        """Gaps the current template can fill, after the terminal and
        residue-name checks. Returns (repairables, skipped) where skipped maps
        each gap rejected for differing residue names to its mismatches."""
        repairables = self.get_repair_residues_from_template(
            rmsd_threshold=rmsd_threshold
        )
        if self.unanchored_gaps:
            listed = ", ".join(
                f"{min(g)}-{max(g)}" for g in sorted(self.unanchored_gaps, key=min)
            )
            warnings.warn(
                f"Template {structure_id}: no residues in common with the model "
                f"chain around gap {listed}; skipped for this template."
            )
        seq_len = len(self.model_chain.can_seq)
        if not include_terminal:
            for gap in [g for g in repairables if 1 in g or seq_len in g]:
                repairables.pop(gap)
            self.model_can_seq = self.model_chain.can_seq

        skipped = {}
        for gap in list(repairables):
            info = repairables[gap]
            mismatches = self._gap_name_mismatches(gap, info["residues"])
            if mismatches and not allow_mismatch:
                skipped[gap] = mismatches
                repairables.pop(gap)
                if warn_mismatch:
                    listed = ", ".join(
                        f"{e}{i} (template: {t})" for i, e, t in mismatches
                    )
                    warnings.warn(
                        f"Gap {min(gap)}-{max(gap)} not built: the template's "
                        f"residues differ from the sequence at {listed}. Use "
                        "allow_mismatch=True to copy them anyway."
                    )
                continue
            bfactors = [
                res["CA"].get_bfactor() for res in info["residues"] if "CA" in res
            ]
            info["name_mismatches"] = mismatches
            info["structure_id"] = structure_id
            info["template_chain"] = getattr(self.template_chain, "id", None)
            info["bfactor_label"] = bfactor_label
            info["mean_bfactor"] = (
                round(float(np.mean(bfactors)), 1) if bfactors else None
            )
        return repairables, skipped

    def _apply_repairables(self, repairables):
        """Copy the chosen gap residues into the model chain and report them."""
        new_items = []
        for gap_ids, info in sorted(repairables.items()):
            self._copy_gap_residues(gap=gap_ids, residues=info["residues"])
            self._accept_variants(info["name_mismatches"])
            template = info["structure_id"]
            if info.get("template_chain") and not str(template).endswith(".cif"):
                template = f"{template}.{info['template_chain']}"
            item = {
                "gap": (min(gap_ids), max(gap_ids)),
                "n_residues": len(gap_ids),
                "template": template,
                "local_rmsd": round(float(info["rmsd"]), 3),
                "bfactor_label": info["bfactor_label"],
                "mean_bfactor": info["mean_bfactor"],
                "junction_CN": self._junction_distances(gap_ids),
                "name_mismatches": info["name_mismatches"],
            }
            self.build_report.append(item)
            new_items.append(item)
        self.model_chain.reset_atom_serial_numbers()
        self.repaired_residues.update(repairables)
        stretched = [
            f"{item['gap'][0]}-{item['gap'][1]} ({side} {dist:.1f} Å)"
            for item in new_items
            for side, dist in item["junction_CN"].items()
            if dist > self.junction_warn
        ]
        if stretched:
            warnings.warn(
                "Copied gaps do not join the chain: C-N distance above "
                f"{self.junction_warn} Å at {', '.join(stretched)}. The template "
                "differs from the model around these gaps; the chain needs loop "
                "closure or restrained minimization before simulation."
            )

    def _accept_variants(self, name_mismatches):
        """Record copied residues whose names differ from the sequence (copied
        with allow_mismatch=True) in reported_res and can_seq, as merge_chains
        does for mismatches, so missing_res no longer lists them."""
        if not name_mismatches:
            return
        chain = self.model_chain
        # Entity.copy() shares this list with the chain the builder was given
        chain.reported_res = list(chain.reported_res)
        index = {resseq: i for i, (resseq, _) in enumerate(chain.reported_res)}
        can_seq = list(str(chain.can_seq))
        for resseq, _expected, resname in name_mismatches:
            name = "HIS" if resname in ("HSD", "HSE", "HSP") else resname
            chain.reported_res[index[resseq]] = (resseq, name)
            can_seq[resseq - 1] = chain.letter_3to1_dict.get(name, "X")
        chain.can_seq = Seq("".join(can_seq))
        self.model_can_seq = chain.can_seq

    def _junction_distances(self, gap_ids):
        """C-N distances (Å) where a copied gap joins the residues before and after it."""
        out = {}
        first, last = min(gap_ids), max(gap_ids)
        for label, (i, j) in {
            "before": (first - 1, first),
            "after": (last, last + 1),
        }.items():
            if i in self.model_chain and j in self.model_chain:
                res_i, res_j = self.model_chain[i], self.model_chain[j]
                if "C" in res_i and "N" in res_j:
                    dist = np.linalg.norm(res_i["C"].coord - res_j["N"].coord)
                    out[label] = round(float(dist), 2)
        return out

    def report(self, bond_warn=None):
        """Print one line per built gap; junctions longer than bond_warn (Å) are flagged."""
        if bond_warn is None:
            bond_warn = self.junction_warn
        if not self.build_report:
            print("No gaps built yet.")
            return
        for item in self.build_report:
            start, end = item["gap"]
            junctions = ", ".join(
                f"{side} {dist:.2f} Å" + (" (stretched)" if dist > bond_warn else "")
                for side, dist in item["junction_CN"].items()
            )
            bfac = (
                f"mean {item['bfactor_label']} {item['mean_bfactor']}"
                if item["mean_bfactor"] is not None
                else ""
            )
            print(
                f"{start}-{end} ({item['n_residues']} res) from {item['template']}: "
                f"local RMSD {item['local_rmsd']:.2f} Å, {bfac}; C-N junction {junctions}"
            )

    def get_chain(self, return_incomplete_chain=False):
        """If no missing loop exists or return_incomplete_chain is True, this
        function returns the chain that is being modelled. Otherwise,
        it raises an error."""
        if not self.model_chain.is_continuous() and not return_incomplete_chain:
            raise ValueError(
                "The chain being worked on remains broken! If you have tried to repair the missing loops, "
                "this means that the method(s) failed on at least one missing loops."
            )

        return self.model_chain


class ArcLoopBuilder:
    """
    Simple loop modeller that builds missing loops as arcs avoiding clashes
    with existing protein atoms.
    """

    def __init__(self, loop, protein_atoms):
        """
        start_residue: dict {'C': np.array, 'CA': np.array} of the residue BEFORE loop
        end_residue:   dict {'N': np.array, 'CA': np.array} of the residue AFTER loop
        sequence:      string of amino acids in the missing loop
        protein_atoms: list of np.array coordinates (the 'walls' to avoid)
        """
        self.topo = ResidueTopologySet("protein")
        self.start = loop["anchor"]
        self.end = loop["target"]
        self.loop = loop["loop"]
        self.protein_atoms = protein_atoms
        self.num_residues = len(loop["loop"])
        self.res_ids = loop["loop_res_ids"]
        # Standard geometries for approximations
        self.CA_CA_DIST = 3.8  # Avg distance between Alpha Carbons
        self.CLASH_DIST = 1.8  # Min distance to existing atoms

    def get_bezier_point(self, t, p0, p1, p2):
        """
        Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        """
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

    def generate_ca_trace(self, apex_point):
        """
        Generates C-alpha coordinates along the Bezier curve defined by
        Start, Apex, and End.
        """
        p0 = self.start["CA"]
        p2 = self.end["CA"]
        p1 = apex_point

        ca_coords = []
        # We need N+2 points (Start + N loop residues + End) to interpolate correctly
        # We only return the N loop residues.
        steps = self.num_residues + 1

        for i in range(1, steps):
            t = i / steps
            coord = self.get_bezier_point(t, p0, p1, p2)
            ca_coords.append(coord)

        return ca_coords

    def reconstruct_backbone(self, ca_trace):
        """
        Simple heuristic to place N and C atoms relative to the C-alphas.
        Since this is an *initial* conformation, we place them on the vectors
        connecting the C-alphas.
        """
        full_loop = []

        # Include anchors for vector calculation
        all_cas = [self.start["CA"]] + ca_trace + [self.end["CA"]]

        for i in range(self.num_residues):
            # Indices in all_cas: Current residue is i+1
            prev_ca = all_cas[i]
            curr_ca = all_cas[i + 1]
            next_ca = all_cas[i + 2]

            # Place N approx 1/3 way from Prev_CA to Curr_CA
            vec_n = curr_ca - prev_ca
            n_coord = prev_ca + vec_n * 0.76  # 0.76 is roughly (dist(CA-N) / 3.8)

            # Place C approx 1/3 way from Curr_CA to Next_CA
            vec_c = next_ca - curr_ca
            c_coord = curr_ca + vec_c * 0.39  # 0.39 is roughly (dist(CA-C) / 3.8)

            coord_dict = {
                "N": np.array(n_coord),
                "CA": np.array(curr_ca),
                "C": np.array(c_coord),
            }
            resname = self.loop[i]
            residue = topo[resname].create_residue_from_coord_dict(
                coord_dict, resseq=self.res_ids[i]
            )
            full_loop.append(residue)

        return full_loop

    def check_clashes(self, backbone):
        """Returns True if any atom in the new backbone hits the protein."""
        for res in backbone:
            for atom_type in ["N", "CA", "C"]:
                coord = res[atom_type].coord
                for p_atom in self.protein_atoms:
                    if np.linalg.norm(coord - p_atom) < self.CLASH_DIST:
                        return True
        return False

    def build(self):
        # 1. Define the Axis (Chord)
        start_pt = self.start["CA"]
        end_pt = self.end["CA"]
        chord_vec = end_pt - start_pt
        chord_len = np.linalg.norm(chord_vec)
        midpoint = start_pt + (chord_vec / 2.0)

        # 2. Determine Loop Height
        # The loop must be longer than the chord.
        # Approx length of loop = Num_Residues * 3.8
        target_len = self.num_residues * self.CA_CA_DIST

        if target_len < chord_len:
            print("Warning: Sequence too short to bridge gap. Stretching straight.")
            h = 0.1
        else:
            # Approx height calculation (Pythagoras on the triangle approximation)
            # This is a heuristic estimation for Bezier control point height
            h = np.sqrt((target_len / 2.0) ** 2 - (chord_len / 2.0) ** 2) * 1.5

        # 3. Define Perpendicular Vectors for Rotation
        # We need an arbitrary vector perpendicular to the chord to start rotating
        # Try Z-axis, if chord is Z-axis, try X-axis
        arbitrary = np.array([0, 0, 1])
        if np.allclose(chord_vec / chord_len, arbitrary):
            arbitrary = np.array([1, 0, 0])

        # Gram-Schmidt process to get a true perpendicular vector
        perp_vec = np.cross(chord_vec, arbitrary)
        perp_vec = perp_vec / np.linalg.norm(perp_vec)

        # 4. Swing the Loop (Rotate 0 to 360 degrees)
        best_backbone = None
        min_clashes = float("inf")

        # We test 36 angles (every 10 degrees)
        for angle in np.linspace(0, 2 * np.pi, 36):
            # Rotate the perpendicular vector around the chord vector
            # (Rodrigues' rotation formula or simple trig reconstruction)

            # Simplified rotation logic:
            # Create a local coordinate system at the midpoint
            # u = chord (normalized), v = perp, w = cross(u, v)
            u = chord_vec / chord_len
            v = perp_vec
            w = np.cross(u, v)

            # Rotated direction
            rot_dir = v * np.cos(angle) + w * np.sin(angle)

            # The "Apex" (Control Point) for the Bezier curve
            apex = midpoint + (rot_dir * h)

            # Generate the trace
            ca_trace = self.generate_ca_trace(apex)

            # Build full atoms
            backbone = self.reconstruct_backbone(ca_trace)

            # Check collisions
            if not self.check_clashes(backbone):
                print(f"Found clear path at angle {int(np.degrees(angle))} degrees!")
                return backbone

        print("Could not find clash-free path. Returning best guess (0 degrees).")
        # In a real scenario, you might return the path with minimum clashes
        # Re-generate base case
        apex = midpoint + (perp_vec * h)
        return self.reconstruct_backbone(self.generate_ca_trace(apex))
