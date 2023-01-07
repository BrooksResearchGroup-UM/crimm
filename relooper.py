import json
import requests
import warnings
from Chain import Chain, PolymerChain
from ChainSuperimposer import ChainSuperimposer
from create_structure import make_schain

class ChainLooper:
    """
    loop modeller for PDB protein structures by homology modeling
    """

    def __init__(self, model_chain: PolymerChain):
        self.model_chain = model_chain
        self.model_can_seq = model_chain.can_seq
        self.repaired_gaps = []
        self.imposer =  ChainSuperimposer()
        
        if self.model_can_seq == None:
            raise AttributeError('Canonical sequence is required to model loops')

    def set_template_chain(self, template_chain: PolymerChain):
        self.template_chain = template_chain
        self.template_can_seq = template_chain.can_seq
        if self.template_can_seq == None:
            raise AttributeError('Canonical sequence is required to model loops')
        else:
            self.model_aligned, self.p_aligned =\
                self.imposer.find_all_common_res(self.model_chain, self.template_chain)
        
        self.repairable = self.find_repairable_gaps()

    def find_repairable_gaps(self):
        if not hasattr(self, 'template_chain'):
            raise AttributeError('Template chain not set!' 
                'Use set_template_chain() to load template first')
        
        repairables = []
        i, j = 0, 0

        while i<len(self.model_chain.gaps) and j<len(self.template_chain.gaps):
            # gap of the model chain
            m_gap = self.model_chain.gaps[i]
            # start idx of the gap
            m_start = min(m_gap)
            # gap of the template chain
            t_gap = self.template_chain.gaps[j]
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
            # move on to the next p_gap (e.g 3)
            elif m_start > t_end:
                j += 1
            # Otherwise, current model_gap can be fixed by the template_chain,
            # and move on to the next model_gap (e.g. 4)
            else:
                repairables.append(m_gap)
                i += 1

        # If there are left-over gaps from receiving chain after going over 
        # the last gap of providing chain, these gaps can all be repaired.
        if i < len(self.model_chain.gaps):
            repairables.extend(self.model_chain.gaps[i:])
        return repairables

    def superimpose_two_chains(self, on_atoms = 'backbone'):
        """
        Impose template chain onto modeling chain
        Return the RMSD value
        """

        if not hasattr(self, 'template_chain'):
            raise AttributeError('Template chain not set!' 
                'Use set_template_chain() to load template first')
        
        self.imposer.set_chains(self.model_chain, self.template_chain, on_atoms=on_atoms)
        self.imposer.apply_transform(self.template_chain)
        return self.imposer.rms

    def _copy_gap_residues(self, gap, keep = False):
        id_st = 1
        # gap is a set and has to be sorted before calculated the idx for 
        # insertion. Otherwise, it would mess up the residue sequence in
        # the structure
        if keep:
            # The residues from the template, template_chain, will get 
            # recursively copied, and template_chain will remain intact. 
            for i in sorted(gap):
                copy_res = self.template_chain[i].copy()
                self.model_chain.insert(i-id_st, copy_res)
        else:
            # By default, we are not using copy() 
            # The caveat is that the gap residues will be directly transfered 
            # from the template structure to the model structure.
            # That is, model_chain will get fixed, but the resulting structure 
            # of template_chain will have the gaps.
            for i in sorted(gap):
                self.model_chain.insert(i-id_st, self.template_chain[i])
                self.template_chain.detach_child(self.template_chain[i].get_id())

    def repair_gaps_from_template(self, 
            template_chain : PolymerChain = None, 
            keep_template_structure = False, 
            rmsd_threshold = 0.8):
        """
        Automatically repair the model_chain from template_chain based on all the repairable 
        gaps.
        """
        if template_chain:
            self.set_template_chain(template_chain=template_chain)
        if not hasattr(self, 'template_chain'):
            raise AttributeError('Template chain not set!' 
                'Use set_template_chain() to load template first')
        
        cur_repaired = []
        for gap in self.find_repairable_gaps():
            st, end = min(gap), max(gap)
            self.imposer.set_around_gap(self.model_chain, self.template_chain, st, end)
            self.imposer.apply_transform(self.template_chain)
            rmsd = self.imposer.rms
            if rmsd <= rmsd_threshold:
                self._copy_gap_residues(gap, keep_template_structure)
                cur_repaired.append(gap)
            else:
                warnings.warn(
                    'RMSD ({}) of superposition around gap {} is '
                    'higher than the threshold {}. Repair not applied!'
                    .format(rmsd, gap, rmsd_threshold))
            self.model_chain.update()
            self.template_chain.update()

        self.model_chain.reset_atom_serial_numbers()
        self.template_chain.reset_atom_serial_numbers()
        self.repaired_gaps.extend(cur_repaired)
        return cur_repaired

    def highlight_repaired_gaps(self, gaps = None, chain = None, add_licorice = False):
        """
        Highlight the repaired gaps with red color and show licorice 
        representations
        """
        ## FIXME: refactor these codes
        if gaps == None:
            gaps = self.repaired_gaps
        if chain == None:
            chain = self.model_chain
        try:
            import nglview as nv
        except ImportError:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            )
        view = nv.NGLWidget()
        blob = self.model_chain.get_pdb_str()
        ngl_args = [{'type': 'blob', 'data': blob, 'binary': False}]
        view._ngl_component_names.append('Model Chain')
        # Load data, and do not add any representation
        view._remote_call("loadFile",
                        target='Stage',
                        args=ngl_args,
                        kwargs= {'ext':'pdb',
                        'defaultRepresentation':False}
                        )
        # Add color existing residues grey
        view._remote_call('addRepresentation',
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
            for res_id in gap:
                atom_ids = [atom.get_serial_number() for atom in self.model_chain[res_id]]
                gap_atom_selection.extend(atom_ids)

        if len(gap_atom_selection) == 0:
            warnings.warn('No repaired gap exists/provided for highlighting!')

        # Convert to string array for JS
        sele_str = "@" + ",".join(str(s) for s in gap_atom_selection)
        # Highlight the repaired gap atoms with red color
        view._remote_call('addRepresentation',
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
            view._remote_call('addRepresentation',
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
        except ImportError:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            )
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
        view._remote_call('autoView',
                        target='compList',
                        args=[sele_str, 0],
                        kwargs={'component_index': 0},
                        )
        return view

    def show(self):
        try:
            import nglview as nv
        except ImportError:
            raise ImportError(
                "WARNING: nglview not found! Install nglview to show"
                "protein structures."
                "http://nglviewer.org/nglview/latest/index.html#installation"
            )
        view = nv.NGLWidget()
        self.model_chain.load_nglview(view) 
        self.template_chain.load_nglview(view)
        return view
    
class ReLooper(ChainLooper):
    def __init__(self, model_chain: PolymerChain, canonical_seq = None) -> None:
        if canonical_seq != None:
            # Overwrite canonical sequence in model chain
            model_chain.can_seq = canonical_seq
        elif model_chain.can_seq == None:
            raise ValueError('Canonical sequence is required to for homology'
                'loop modeling!')
        super().__init__(model_chain)
        self.query_results = None

    def _build_seq_query(
            self, seq_string, max_num_match, identity_score_cutoff
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
        seq_string = str(self.model_chain.can_seq)
        json_q = self._build_seq_query(seq_string, 
            max_num_match, identity_score_cutoff)
        # entry point of rcsb.org for advancced search
        url = "https://search.rcsb.org/rcsbsearch/v2/query?"
        r = requests.post(url, json=json_q)
        r_dict = json.loads(r.text)
        if 'result_set' in r_dict:
            results = dict()
            for entry in r_dict['result_set']:
                pdb_id, chain_id = entry['identifier'].split('.')
                if entry['score'] not in results:
                    results[entry['score']] = dict()
                if pdb_id not in results[entry['score']]:
                    results[entry['score']][pdb_id] = []
                results[entry['score']][pdb_id].append(chain_id)
            return results
        else:
            raise ValueError('Query on the sequence did not return '+\
            'requested information. {}'.format(r))

    def get_templates(self, query_results: dict):
        """
        Return a generator of all template chains from the query result dict
        """
        for entity_dict in query_results.values():
            for pdbid, chain_ids in entity_dict.items():
                for chain_id in chain_ids:
                    template_chain = make_schain(pdbid, chain_id)
                    # we want to use a generator here because the loop for
                    # auto_rebuild has a breaking condition, and pulling 
                    # all PDB template model data from rcsb.org at once 
                    # could create more overhead
                    yield (pdbid, template_chain)

    def _reject_by_rmsd(self, overall_rmsd_threshold):
        if overall_rmsd_threshold != None:
            rms = self.superimpose_two_chains()
            if rms > overall_rmsd_threshold:
                return True
        return False
            
    def auto_rebuild(
            self, 
            max_num_match = 20,
            identity_score_cutoff = 0.95,
            rmsd_threshold = 0.8, 
            overall_rmsd_threshold = None):
        
        self.query_results = self.query_seq_match(max_num_match,
            identity_score_cutoff)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # create a generator for all the templates
            all_templates = self.get_templates(self.query_results)

        all_repaired = []
        for pdbid, template in all_templates:
            self.set_template_chain(template)
            if self._reject_by_rmsd(overall_rmsd_threshold):
                continue
            cur_repaired = self.repair_gaps_from_template(
                rmsd_threshold = rmsd_threshold
                )
            if cur_repaired:
                all_repaired.append((pdbid, cur_repaired))
            if self.model_chain.is_continuous(): 
                break
        return all_repaired