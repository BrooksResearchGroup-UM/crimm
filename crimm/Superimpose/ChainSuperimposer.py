
import warnings
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import Superimposer
from crimm.StructEntities.Chain import PolymerChain
from crimm.Visualization.NGLVisualization import show_nglview_multiple

##TODO: Refactor this!!
class ChainSuperimposer(Superimposer):
    """Superimpose two chains using their canonical sequences if available"""
    def __init__(self) -> None:
        super().__init__()
        self.ref_chain: PolymerChain = None
        self.mov_chain: PolymerChain = None
        self.aligner = PairwiseAligner()
        # We don't want highly fragmented alignments
        self.aligner.target_internal_open_gap_score = -100
        self.aligner.query_internal_open_gap_score = -100
        self.alignments = None
        self.ref_ranges, self.mov_ranges = None, None

    def _check_chain_type(self, chain):
        if not isinstance(chain, PolymerChain):
            raise TypeError(
                "ChainSuperimposer only works with PolymerChain class"
            )
        
    def get_matching_res(self, ref_chain, mov_chain):
        """ Find all aligned residues that exist in both chains """
        self.ref_chain = ref_chain
        self.mov_chain = mov_chain
        if self.ref_chain.can_seq == self.mov_chain.can_seq:
            # If canonical sequences are identical
            # Use all present residues from both chains
            self.alignments = ["identical sequences"]
            complete_range = range(1, len(self.ref_chain.can_seq)+1)
            id_pairs = zip(complete_range, complete_range)

        else:
            # Canonical sequences are not identical
            # Align them first to find the common segments
            self.alignments = self.aligner.align(
                self.ref_chain.can_seq, self.mov_chain.can_seq
            )
            top_alignment = self.alignments[0]
            self.ref_ranges, self.mov_ranges = top_alignment.aligned
            res_ids = top_alignment.indices+1 # residue seq ids start from 1
            aligned_ids = np.logical_and(res_ids[0], res_ids[1])
            id_pairs = res_ids.T[aligned_ids]

        r_res, m_res = self._get_aligned_res(id_pairs)
        return r_res, m_res

    def _get_aligned_res(self, id_pairs):
        """
        Get a set of residue ids from the aligned ranges
        """
        ref_aligned_res = []
        mov_aligned_res = []
        for ref_id, mov_id in id_pairs:
            # needs to be converted to int because the ids are numpy.int64
            ref_id, mov_id = int(ref_id), int(mov_id)
            if ref_id in self.ref_chain and mov_id in self.mov_chain:
                ref_res = self.ref_chain[ref_id]
                mov_res = self.mov_chain[mov_id]
                if ref_res.resname != mov_res.resname:
                    warnings.warn(
                        "Residues are not identical at "
                        f"model chain: {ref_res.resname}-{ref_res.id[1]} and "
                        f"template chain: {mov_res.resname}-{mov_res.id[1]}"
                    )
                    continue
                ref_aligned_res.append(ref_res)
                mov_aligned_res.append(mov_res)

        return ref_aligned_res, mov_aligned_res

    def set_chains(self, ref_chain, mov_chain, on_atoms = 'CA'):
        """Set the chains to be superimposed. mov_chain will be moved to
        ref_chain."""

        self._check_chain_type(ref_chain)
        self._check_chain_type(mov_chain)

        seq_len = len(ref_chain.can_seq)
        # We don't want highly fragmented alignments. Gap opening is only allowed 
        # in the middle of the sequence when 20% or more of the sequence length 
        # is aligned as a result of that
        self.aligner.target_internal_open_gap_score = -(seq_len // 5)
        self.aligner.query_internal_open_gap_score = -(seq_len // 5)

        ref_res, mov_res = self.get_matching_res(ref_chain, mov_chain)

        if on_atoms == 'CA':
            ref_align_atoms, mov_align_atoms = \
                    self._find_common_CA_atoms(ref_res, mov_res)
        elif on_atoms == 'backbone':
            ref_align_atoms, mov_align_atoms = \
                    self._find_common_backbone_atoms(ref_res, mov_res)
        elif on_atoms == 'all':
            ref_align_atoms, mov_align_atoms = \
                    self._find_all_common_atoms(ref_res, mov_res)
        else:
            raise ValueError('on_atoms has to be selected from '
                '{"backbone", "all", "CA"}') 
        self.set_atoms(ref_align_atoms, mov_align_atoms)

    def _find_all_common_atoms(self, ref_res_list, mov_res_list):
        """Find all common atoms from two lists of residues that are aligned"""
        ref_align_atoms = []
        mov_align_atoms = []
        for ref_res, mov_res in zip(ref_res_list, mov_res_list):
            for atom_id in ref_res.child_dict.keys():
                # Make sure move chain residues have the same atoms
                # Otherwise, the missing atom will be excluded
                if atom_id in mov_res:
                    ref_align_atoms.append(ref_res[atom_id])
                    mov_align_atoms.append(mov_res[atom_id])

        return ref_align_atoms, mov_align_atoms

    def _find_common_CA_atoms(self, ref_res_list, mov_res_list):
        ref_align_atoms = []
        mov_align_atoms = []
        for ref_res, mov_res in zip(ref_res_list, mov_res_list):
            # Make sure both residues have the CA atom
            # Otherwise, the missing atom will be excluded
            if 'CA' in ref_res and 'CA' in mov_res:
                ref_align_atoms.append(ref_res['CA'])
                mov_align_atoms.append(mov_res['CA'])

        return ref_align_atoms, mov_align_atoms

    def _find_common_backbone_atoms(self, ref_res_list, mov_res_list):
        ref_align_atoms = []
        mov_align_atoms = []
        for ref_res, mov_res in zip(ref_res_list, mov_res_list):
            for atom_id in ['N','CA','C','O']:
                # Make sure both residues have the backbone atom
                # Otherwise, the missing atom will be excluded
                if atom_id in ref_res and atom_id in mov_res:
                    ref_align_atoms.append(ref_res[atom_id])
                    mov_align_atoms.append(mov_res[atom_id])

        return ref_align_atoms, mov_align_atoms

    def find_valid_edge_ids(self, gap_ids, avail_res, cutoff):
        """Find the residue id range that is present around the gap."""
        avail_ids = [res.id[1] for res in avail_res]
        left, right = min(gap_ids), max(gap_ids)
        res_ids = []
        self._recur_find_ids(avail_ids, cutoff, left, res_ids, -1)
        self._recur_find_ids(avail_ids, cutoff, right, res_ids, 1)
        return sorted(res_ids)

    @staticmethod
    def _recur_find_ids(avail_ids, remainder, cur_id, valid_ids, direction):
        for i in range(1, remainder+1):
            # forward_direction == 1, backward == -1
            cur_id += 1*direction
            if cur_id < 1 or cur_id > max(avail_ids):
                # inclusive on both edges: 1 to max(avail_ids)
                return
            if cur_id in avail_ids:
                valid_ids.append(cur_id)
            else:
                ChainSuperimposer._recur_find_ids(
                    avail_ids, remainder+1-i, cur_id, valid_ids, direction
                )
                return

    def set_around_gap(
            self, ref_chain, mov_chain,
            ref_gap,
            cutoff=10
        ):

        """
        Set atoms for superposition around a gap/missing residues on the ref 
        chain. Start and end residue index needed. Alignment will be performed on
        backbone atoms. If mov_gap is not specified, identical sequence will be 
        assumed on both chains, and the reference chain gap locations will be applied
        on the move chain 
        """
        ## TODO: add options for on_atoms

        self._check_chain_type(ref_chain)
        self._check_chain_type(mov_chain)
        # Get all matched residues from alignment
        all_ref_res, all_mov_res = self.get_matching_res(ref_chain, mov_chain)
        # Find the existing residue ids that are within the cutoff range
        ref_res_ids = self.find_valid_edge_ids(ref_gap, all_ref_res, cutoff)

        ref_edge_res = []
        mov_edge_res = []
        for ref_res, mov_res in zip(all_ref_res, all_mov_res):
            if ref_res.id[1] in ref_res_ids:
                ref_edge_res.append(ref_res)
                mov_edge_res.append(mov_res)

        if ref_edge_res is None:
            warnings.warn('The cutoff value extends out of the available '
                    'residues in the chain. All backbone atoms on both chains '
                    'are selected for superposition')
            ref_edge_res, mov_edge_res = all_ref_res, all_mov_res

        ref_atoms, mov_atoms = self._find_common_backbone_atoms(
            ref_edge_res, mov_edge_res
        )
        self.set_atoms(ref_atoms, mov_atoms)
    
    def apply_transform(self, entity: PolymerChain=None):
        if entity is None:
            entity = self.mov_chain
        self.apply(entity.get_atoms())

    def show(self):
        return show_nglview_multiple([self.ref_chain, self.mov_chain])