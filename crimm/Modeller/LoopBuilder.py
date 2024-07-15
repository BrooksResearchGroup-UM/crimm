import json
import warnings
import requests
from Bio.Align import PairwiseAligner
from crimm.Superimpose.ChainSuperimposer import ChainSuperimposer
from crimm.Fetchers import fetch_rcsb, fetch_alphafold, uniprot_id_query
import crimm.StructEntities as Entities

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
    chainA_aligned, chainB_aligned = align.aligned+1 # resseq is 1-indexed
    offsets = (chainB_aligned - chainA_aligned)[:,0]
    offset_dict = {
        range(*segmentA): offset for segmentA, offset in zip(
            chainA_aligned, offsets
        )
    }
    return offset_dict

def find_gap_offsets(gaps, segment_offset_dict):
    gap_offset_dict = {}
    for segment, offset in segment_offset_dict.items():
        segment = set(segment)
        gaps_in_range = find_gaps_within_range(gaps, segment)
        gap_offset_dict.update(
            {tuple(gap): offset for gap in gaps_in_range}
        )
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
        translated = {
            tuple(sorted(gap)):tuple(sorted(gap)) for gap in rp_gap
        }
        return translated

    segment_offsets = find_segment_offsets(chainA.can_seq, chainB.can_seq)
    gap_offsets = find_gap_offsets(chainA.gaps, segment_offsets)
    translated = {}
    for gap_A, offset in gap_offsets.items():
        translated_gap_A = {resseq+offset for resseq in gap_A}
        if find_gaps_range_overlap(chainB.gaps, translated_gap_A):
            continue
        translated[tuple(sorted(gap_A))] = tuple(sorted(translated_gap_A))
    return translated

def translate_gaps_with_identical_seqs(chainA, chainB):
    model_chain_gaps = chainA.gaps
    template_chain_gaps = chainB.gaps
    translated = []
    i, j = 0, 0

    while i<len(model_chain_gaps) and j<len(template_chain_gaps):
        # gap of the model chain
        m_gap = model_chain_gaps[i]
        # start idx of the gap
        m_start = min(m_gap)
        # gap of the template chain
        t_gap = template_chain_gaps[j]
        # end idx of the gap
        t_end =  max(t_gap)
             
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

class ChainLoopBuilder:
    """
    loop modeller for PDB protein structures by homology modeling
    """

    def __init__(
            self, model_chain: Entities.PolymerChain, 
            model_can_seq = None,
            pdbid = None
        ):
        self.model_chain = model_chain.copy()
        if model_can_seq is not None:
            self.model_can_seq = model_can_seq
        else:
            self.model_can_seq = model_chain.can_seq
        self.repaired_gaps = []
        self.repaired_residues = {}
        self.imposer =  ChainSuperimposer()
        self.template_chain = None
        self.template_can_seq = None
        self.translated_ids = None
        self.query_results = None
        self.pdbid = pdbid
        if self.pdbid is None:
            if model_chain.get_top_parent().level == 'S':
                self.pdbid = model_chain.get_top_parent().id
            else:
                warnings.warn(
                    f'PDB ID not set for {model_chain}! '
                    'Make sure the chain is loaded from a MMCIF file or from fetch() '
                    'to have the PDB ID correctly parsed, or set the PDB ID manually.'
                )

        if self.model_can_seq is None:
            raise AttributeError('Canonical sequence is required to repair loops')

    def set_template_chain(
            self, template_chain: Entities.PolymerChain
        ):
        self.template_chain = template_chain
        self.template_can_seq = template_chain.can_seq
        if self.template_can_seq is None:
            raise AttributeError(
                f'Canonical sequence does not exist for {template_chain}! '
                'Make sure the chain is loaded from a MMCIF file or from fetch() '
                'to have the canonical sequence correctly parsed.'
            )

        self.translated_ids = translate_gap_ids(
            self.model_chain, self.template_chain
        )

    def superimpose_two_chains(self, on_atoms = 'backbone'):
        """
        Impose template chain onto modeling chain
        Return the RMSD value
        """

        if not hasattr(self, 'template_chain'):
            raise AttributeError(
                'Template chain not set! Use set_template_chain() to load template first'
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
            res.id = (hetflag, i, ' ')
            # i in insert refer to child list index, thus use i-1 here
            self.model_chain.add(res)
        self.model_chain.sort_residues()
        self.repaired_gaps.append(tuple(gap))

    def get_repair_residues_from_template(
            self,
            rmsd_threshold = None,
        ):
        """
        Find repairable residue for model_chain from template_chain gaps.
        """
        if not hasattr(self, 'template_chain'):
            raise AttributeError(
                'Template chain not set!' 
                'Use set_template_chain() to load template first'
            )

        rmsd_qualified = {}
        for gap, template_resseqs in self.translated_ids.items():
            self.imposer.set_around_gap(
                self.model_chain, self.template_chain, gap, cutoff=5
            )
            self.imposer.apply_transform(self.template_chain)
            rmsd = self.imposer.rms
            if rmsd_threshold is not None and rmsd > rmsd_threshold:
                warnings.warn(
                    f'RMSD ({rmsd}) of superposition around gap {gap} is '
                    f'higher than the threshold {rmsd_threshold}. '
                    'Repair not applied!'
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

            rmsd_qualified[gap] = {
                "residues":available_residues,
                'rmsd':rmsd
            }

        return rmsd_qualified

    def highlight_repaired_gaps(self, color = 'green', add_licorice = False):
        """
        Highlight the repaired gaps with color (default: cyan)
        """
        try:
            from crimm.Visualization.NGLVisualization import View
        except ImportError as exc:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            ) from exc
        fixed_res = []
        for res_list in self.repaired_gaps:
            for res_seq in res_list:
                fixed_res.append(self.model_chain[res_seq])
        if len(fixed_res) < 5:
            add_licorice = True # cartoon will not render if there are too few residues
        view = View()
        view.load_entity(self.model_chain)
        view.highlight_residues(
            fixed_res, color=color, add_licorice=add_licorice
        )

        return view

    def show_gaps(self):
        """
        Highlight the residues on the gap edges with color (default: red)
        """
        if len(self.model_chain.gaps) == 0:
            warnings.warn(f"No gap found in {self.model_chain}!")
        try:
            from crimm.Visualization.NGLVisualization import View
        except ImportError as exc:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            ) from exc
        view = View()

        view.load_entity(self.model_chain)
        view.subdue_all_entities()
        rgb_red = [1,0,0]
        line_width = 0.5
        for gap in self.model_chain.gaps:
            # Center on the gap by selecting the terminal residues around the gap
            if (
                (st:=min(gap)-1) in self.model_chain and 
                (end:=max(gap)+1) in self.model_chain
            ):
                st_coord = self.model_chain[st]['CA'].coord
                end_coord = self.model_chain[end]['CA'].coord
                arrow_name = f'gap:{st+1}-{end-1}'
                view.shape.add_arrow(
                    st_coord, end_coord, rgb_red, line_width, arrow_name
                )

        return view

    def show(self):
        from crimm.Visualization.NGLVisualization import show_nglview_multiple
        return show_nglview_multiple([self.model_chain, self.template_chain])

    @staticmethod
    def _build_seq_query(
            seq_string, max_num_match, identity_score_cutoff
        ):
        """
        Args:
            seq_string: str
            max_num_match: int
            identity_score_cutoff: float (from 0 to 1)
        """
        query_dict = {
            
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
            "evalue_cutoff": 1,
            "identity_cutoff": identity_score_cutoff,
            "sequence_type": "protein",
            "value": seq_string
            }
            },
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {
                "start": 0,
                "rows": max_num_match
            }
        },
        "return_type": "polymer_instance"
        }
        
        return query_dict

    def query_seq_match(self, max_num_match, identity_score_cutoff):
        '''
        Make RCSB PDB query for protein sequence
        
        :param seq_string : string of the query sequence
        :param max_num_match : maximum number of top matches returned. 
        :param identity_score_cutoff: lowest identity score accepted
        :rtype retults : dict consists (sequence_matching_score, PDB_ID, entity_ID)
        '''
        seq_string = str(self.model_can_seq)
        json_q = ChainLoopBuilder._build_seq_query(
            seq_string, max_num_match, identity_score_cutoff
        )
        # entry point of rcsb.org for advancced search
        url = "https://search.rcsb.org/rcsbsearch/v2/query?"
        response = requests.post(url, json=json_q, timeout=500)
        r_dict = json.loads(response.text)

        if 'result_set' not in r_dict:
            raise ValueError(
                'Query on the sequence did not return '
                f'the requested information. {response}'
            )

        results = {}
        for entry in r_dict['result_set']:
            pdb_id, chain_id = entry['identifier'].split('.')
            if pdb_id == self.pdbid:
                continue
            if entry['score'] not in results:
                results[entry['score']] = {}
            if pdb_id not in results[entry['score']]:
                results[entry['score']][pdb_id] = []
            results[entry['score']][pdb_id].append(chain_id)
        return results

    @staticmethod
    def get_templates(query_results: dict, local_entry_point: str):
        """
        Return a generator of all template chains from the query result dict
        """
        for entity_dict in query_results.values():
            for pdbid, chain_ids in entity_dict.items():
                structure = fetch_rcsb(
                    pdbid, use_bio_assembly = False,
                    include_solvent = False, local_entry=local_entry_point
                )
                first_model = structure.child_list[0]
                for chain_id in chain_ids:
                    template_chain = first_model[chain_id]
                    yield (pdbid, template_chain)

    def _reject_by_rmsd(self, overall_rmsd_threshold):
        if overall_rmsd_threshold is not None:
            rms = self.superimpose_two_chains()
            if rms > overall_rmsd_threshold:
                return True
        return False
            
    def _update_candidates(self, cur_repairables, candidates, pdbid):
        for gap_key, info_dict in cur_repairables.items():
            if gap_key not in candidates or (
                candidates[gap_key]['rmsd'] > info_dict['rmsd']
            ):  
                info_dict['structure_id'] = pdbid
                candidates[gap_key] = info_dict

    def build_from_homology(
            self,
            max_num_match = 10,
            identity_score_cutoff = 0.95,
            rmsd_threshold = None,
            overall_rmsd_threshold = None,
            local_entry_point = None
        ):
        """Build the missing loops of the model chain from homology models from
        PDB. Seqeunce search query is performed first to find the closest homology
        models based on sequence identity scores."""
        # if local_entry_point is None, rcsb.org will be used
        if self.model_chain.is_continuous():
            warnings.warn(
                f"{self.model_chain} does not have any missing loops"
            )
            return

        self.query_results = self.query_seq_match(
            max_num_match, identity_score_cutoff
        )

        if not self.query_results:
            warnings.warn(
                f"No homology model found for {self.pdbid}-{self.model_chain.id} "
                "from sequence search!"
            )
            return

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # create a generator for all the templates
            all_templates = self.get_templates(
                self.query_results, local_entry_point
            )

        repair_candidates = {}
        for pdbid, template in all_templates:
            self.set_template_chain(template)
            if self._reject_by_rmsd(overall_rmsd_threshold):
                continue
            if cur_repairables := self.get_repair_residues_from_template(
                    rmsd_threshold = rmsd_threshold
                ):
                self._update_candidates(cur_repairables, repair_candidates, pdbid)

        for gap in sorted(repair_candidates):
            copy_residues = repair_candidates[gap]['residues']
            self._copy_gap_residues(
                gap = gap,
                residues = copy_residues,
            )
        self.model_chain.reset_atom_serial_numbers()
        self.repaired_residues.update(repair_candidates)
    
    def build_from_alphafold(self, include_terminal = False):
        """Build the missing loops of the model chain from AlphaFold models. 
        The AlphaFold models are downloaded from the AlphaFold database. If 
        include_terminal is True, the missing terminal residues will be copied
        from the AlphaFold model."""
        
        if len(self.model_chain.missing_res) == 0:
            warnings.warn(
                f"{self.model_chain} does not have any missing residues"
            )
            return

        if self.pdbid is None:
            warnings.warn(
                "Missing looper not built due to no PDB ID provided "
                f"for {self.model_chain}."
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
        self.set_template_chain(af_chain)
        repairables = self.get_repair_residues_from_template(
            rmsd_threshold = None
        )

        if len(repairables) == 0:
            warnings.warn(
                f"No repairable residues found for {self.model_chain} "
                f"from AlphaFold model {af_struct.id}."
            )
            return

        if not include_terminal:
            missing_terminals = []
            for gap in repairables:
                if 1 in gap or len(self.model_chain.can_seq) in gap:
                    missing_terminals.append(gap)
            # remove alphafold terminal residues from repairables
            for gap in missing_terminals:
                repairables.pop(gap)
            self.model_can_seq = self.model_chain.can_seq

        for gap_ids, repairable_info_dict in repairables.items():
            repairable_info_dict['structure_id'] = af_struct.id
            copy_residues = repairable_info_dict['residues']
            self._copy_gap_residues(
                gap = gap_ids,
                residues = copy_residues,
            )
        self.model_chain.reset_atom_serial_numbers()
        self.repaired_residues.update(repairables)

    def get_chain(self):
        """Get the chain that is being modelled on"""
        return self.model_chain