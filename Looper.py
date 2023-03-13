import json
from io import StringIO
import warnings
import requests
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.Align import PairwiseAligner
from test_pdb import fetch_local_cif as fetch_cif
from Chain import PolymerChain
from ChainSuperimposer import ChainSuperimposer
from ChMMCIFParser import ChMMCIFParser

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

def find_repairable_gaps(chainA, chainB):
    if chainA.can_seq == chainB.can_seq:
        rp_gap = find_repairable_gaps_with_identical_seqs(chainA, chainB)
        repairables = {
            tuple(sorted(gap)):tuple(sorted(gap)) for gap in rp_gap
        }
        return repairables
    
    segment_offsets = find_segment_offsets(chainA.can_seq, chainB.can_seq)
    gap_offsets = find_gap_offsets(chainA.gaps, segment_offsets)
    repairables = {}
    for gap_A, offset in gap_offsets.items():
        translated_gap_A = {resseq+offset for resseq in gap_A}
        if find_gaps_range_overlap(chainB.gaps, translated_gap_A):
            continue
        repairables[tuple(sorted(gap_A))] = tuple(sorted(translated_gap_A))
    return repairables

def find_repairable_gaps_with_identical_seqs(chainA, chainB):
    model_chain_gaps = chainA.gaps
    template_chain_gaps = chainB.gaps
    repairables = []
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
            repairables.append(m_gap)
            i += 1

    # If there are left-over gaps from receiving chain after going over 
    # the last gap of providing chain, these gaps can all be repaired.
    if i < len(model_chain_gaps):
        repairables.extend(model_chain_gaps[i:])
    return repairables

def get_available_residues(chain, res_seq_ids, include_het):
    available_residues = []
    for i in res_seq_ids:
        if i in chain:
            available_residues.append(chain[i])
        elif include_het and (het_ids := chain.find_het_by_resseq(i)):
            available_residues.append(chain[het_ids[0]])
        else:
            return None
    return available_residues

class ChainLoopBuilder:
    """
    loop modeller for PDB protein structures by homology modeling
    """

    def __init__(self, model_chain: PolymerChain, model_can_seq = None):
        self.model_chain = model_chain.copy()
        if model_can_seq is not None:
            self.model_can_seq = model_can_seq
        else:
            self.model_can_seq = model_chain.can_seq
        self.repaired_gaps = []
        self.imposer =  ChainSuperimposer()
        self.template_chain = None
        self.template_can_seq = None
        self.repairable = None
        self.query_results = None

        if self.model_can_seq is None:
            raise AttributeError('Canonical sequence is required to repair loops')

    def set_template_chain(
            self, template_chain: PolymerChain, template_can_seq = None
        ):
        self.template_chain = template_chain
        if template_can_seq is not None:
            self.template_can_seq = template_can_seq
        else:
            self.template_can_seq = template_chain.can_seq
        if self.template_can_seq is None:
            raise AttributeError('Canonical sequence is required to repair loops')

        self.repairable = find_repairable_gaps(
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

    def _copy_gap_residues(self, gap, residues, keep = False):
        # gap is a set and has to be sorted before calculated the idx for 
        # insertion. Otherwise, it would mess up the residue sequence in
        # the structure

        if keep:
            # The residues from the template, template_chain, will get 
            # recursively copied, and template_chain will remain intact.
            residues = [res.copy() for res in residues]
        # By default, we are not using copy() 
        # This results the gap residues being directly transfered 
        # from the template structure to the model structure.
        # That is, model_chain will get fixed, but the resulting structure 
        # of template_chain will have the gaps.
        residues = sorted(residues, key=lambda x: x.id[1])
        gap = sorted(gap)

        for i, res in zip(gap, residues):
            # i in insert refer to child list index, thus i-1 here
            self.model_chain.insert(i-1, res)
        self.model_chain.update()
        self.repaired_gaps.append(tuple(gap))
    
    def get_repair_residues_from_template(
            self,
            template_chain : PolymerChain = None,
            # keep_template_structure = False,
            rmsd_threshold = None,
            include_het = False
        ):
        """
        Find repairable residue for model_chain from template_chain gaps.
        """
        if template_chain:
            self.set_template_chain(template_chain=template_chain)
        if not hasattr(self, 'template_chain'):
            raise AttributeError(
                'Template chain not set!' 
                'Use set_template_chain() to load template first'
            )

        rmsd_qualified = {}
        for gap, translated_resseq in self.repairable.items():
            self.imposer.set_around_gap(
                self.model_chain, self.template_chain, 
                gap, translated_resseq, cutoff=5
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

            repairable_residues = get_available_residues(
                    self.template_chain, translated_resseq, 
                    include_het=include_het
                )
            if repairable_residues is None:
                continue

            repairables_residues = [
                res.copy() for res in repairable_residues
            ]
            rmsd_qualified[gap] = {
                "residues":repairables_residues,
                'rmsd':rmsd
            }

        # self.model_chain.reset_atom_serial_numbers()
        # self.template_chain.reset_atom_serial_numbers()
        # self.repaired_gaps.extend(rmsd_qualified)
        return rmsd_qualified

    def highlight_repaired_gaps(
            self, gaps = None, chain = None, add_licorice = False
        ):
        """
        Highlight the repaired gaps with red color and show licorice 
        representations
        """
        ## FIXME: refactor these codes
        if gaps is None:
            gaps = self.repaired_gaps
        if chain is None:
            chain = self.model_chain
        try:
            import nglview as nv
        except ImportError as exc:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            ) from exc
        view = nv.NGLWidget()
        blob = self.model_chain.get_pdb_str()
        ngl_args = [{'type': 'blob', 'data': blob, 'binary': False}]
        view._ngl_component_names.append('Model Chain')
        # Load data, and do not add any representation
        view._remote_call(
            "loadFile",
            target='Stage',
            args=ngl_args,
            kwargs= {'ext':'pdb',
            'defaultRepresentation':False}
        )
        # Add color existing residues grey
        view._remote_call(
            'addRepresentation',
            target='compList',
            args=['cartoon'],
            kwargs={
                'sele': 'protein', 
                'color': 'grey', 
                'component_index': 0
            }
        )

        # Select atoms by atom indices
        gap_atom_selection = []
        for gap in gaps:
            for resseq in gap:
                if resseq in chain:
                    res = self.model_chain[resseq]
                elif (het_ids := self.model_chain.find_het_by_resseq(resseq)):
                    res = self.model_chain[het_ids[0]]
                else:
                    continue
                atom_ids = [atom.get_serial_number() for atom in res]
                gap_atom_selection.extend(atom_ids)

        if len(gap_atom_selection) == 0:
            warnings.warn('No repaired gap exists/provided for highlighting!')

        # Convert to string array for JS
        sele_str = "@" + ",".join(str(s) for s in gap_atom_selection)
        # Highlight the repaired gap atoms with red color
        view._remote_call(
            'addRepresentation',
            target='compList',
            args=['cartoon'],
            kwargs={
                'sele': sele_str, 
                'color': 'red', 
                'component_index': 0
            }
        )
        if add_licorice:
            # Add licorice representations
            view._remote_call(
                'addRepresentation',
                target='compList',
                args=['licorice'],
                kwargs={
                    'sele': sele_str,  
                    'component_index': 0
                }
            )
        view.center()
        return view

    def show_gap(self, gap):
        """
        Show protein structure and center on the gap
        """
        try:
            import nglview as nv
        except ImportError as exc:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            ) from exc
        view = nv.NGLWidget()
        self.model_chain.load_nglview(view)
        # The selection is the start and end of residue ids in the for 'st-end'
        sele_str = ''
        # Center on the gap by selecting the terminal residues around the gap
        if min(gap)-1 in self.model_chain:
            st = str(min(gap)-1)
        else:
            st = ''
        if max(gap)+1 in self.model_chain:
            end = str(max(gap)+1)
        else:
            end = ''

        sele_str = '-'.join([st,end]).strip('-')
        if sele_str == '':
            raise ValueError('Invalid selection of gap: {}'.format(gap))
        view._remote_call(
            'autoView',
            target='compList',
            args=[sele_str, 0],
            kwargs={'component_index': 0},
        )
        return view

    def show(self):
        from NGLVisualization import load_nglview_multiple
        return load_nglview_multiple([self.model_chain, self.template_chain])

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

    @staticmethod
    def query_seq_match(sequence, max_num_match, identity_score_cutoff):
        '''
        Make RCSB PDB query for protein sequence
        
        :param seq_string : string of the query sequence
        :param max_num_match : maximum number of top matches returned. 
        :param identity_score_cutoff: lowest identity score accepted
        :rtype retults : dict consists (sequence_matching_score, PDB_ID, entity_ID)
        '''
        seq_string = str(sequence)
        json_q = ChainLoopBuilder._build_seq_query(seq_string, 
            max_num_match, identity_score_cutoff)
        # entry point of rcsb.org for advancced search
        url = "https://search.rcsb.org/rcsbsearch/v2/query?"
        r = requests.post(url, json=json_q, timeout=500)
        r_dict = json.loads(r.text)

        if 'result_set' not in r_dict:
            raise ValueError(
                'Query on the sequence did not return '
                f'the requested information. {r}'
            )

        results = dict()
        for entry in r_dict['result_set']:
            pdb_id, chain_id = entry['identifier'].split('.')
            if entry['score'] not in results:
                results[entry['score']] = dict()
            if pdb_id not in results[entry['score']]:
                results[entry['score']][pdb_id] = []
            results[entry['score']][pdb_id].append(chain_id)
        return results

    @staticmethod
    def get_templates(query_results: dict):
        """
        Return a generator of all template chains from the query result dict
        """
        cif_parser = ChMMCIFParser(
            first_assembly_only = False, include_solvent = False, QUIET=True
        )
        for entity_dict in query_results.values():
            for pdbid, chain_ids in entity_dict.items():
                cif_str = fetch_cif(pdbid)
                file_handle = StringIO(cif_str)
                structure = cif_parser.get_structure(file_handle)
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
                info_dict['PDBID'] = pdbid
                candidates[gap_key] = info_dict
            
    def auto_rebuild(
            self,
            max_num_match = 10,
            identity_score_cutoff = 0.95,
            rmsd_threshold = None,
            overall_rmsd_threshold = None,
            include_het = False
        ):

        if self.model_chain.is_continuous():
            warnings.warn(f"{self.model_chain} does not have any missing residues")
            return

        self.query_results = self.query_seq_match(
            self.model_chain.can_seq, max_num_match, identity_score_cutoff
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # create a generator for all the templates
            all_templates = self.get_templates(self.query_results)

        repair_candidates = {}
        for pdbid, template in all_templates:
            self.set_template_chain(template)
            if self._reject_by_rmsd(overall_rmsd_threshold):
                continue
            if cur_repairables := self.get_repair_residues_from_template(
                    rmsd_threshold = rmsd_threshold,
                    include_het=include_het
                ):
                self._update_candidates(cur_repairables, repair_candidates, pdbid)

        for gap in sorted(repair_candidates):
            copy_residues = repair_candidates[gap]['residues']
            self._copy_gap_residues(
                gap = gap,
                residues = copy_residues,
                keep = False
            )
        self.model_chain.reset_atom_serial_numbers()

        return repair_candidates
    
    def get_chain(self):
        """Get the chain that is being modelled on"""
        return self.model_chain