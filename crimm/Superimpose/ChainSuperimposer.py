
import warnings
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import Superimposer
from crimm.StructEntities.Chain import PolymerChain
from crimm.Visualization.NGLVisualization import show_nglview_multiple

##TODO: Refactor this!!
class ChainSuperimposer(Superimposer):
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

    def get_matching_res(self):
        """ Find all aligned residues that exist in both chains """
        if self.ref_chain.can_seq == self.mov_chain.can_seq:
            # If canonical sequences are identical
            # Use all present residues from both chains
            self.alignments = "identical sequences"
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
                ref_aligned_res.append(self.ref_chain[ref_id])
                mov_aligned_res.append(self.mov_chain[mov_id])

        return ref_aligned_res, mov_aligned_res

    def set_chains(self, ref_chain, mov_chain, on_atoms = 'CA'):
        """Set the chains to be superimposed. mov_chain will be moved to
        ref_chain."""
        # if any of the chains is not Chain class, convert them
        if not (
                isinstance(ref_chain, PolymerChain) and
                isinstance(mov_chain, PolymerChain)
            ):
            raise TypeError(
                "ChainSuperimposer only works with PolymerChain class"
            )
        self.ref_chain = ref_chain
        self.mov_chain = mov_chain

        ref_res, mov_res = self.get_matching_res()

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

    @staticmethod
    def find_valid_residue_range(ids, seq_len, cutoff):
        start, end = min(ids), max(ids)
        # select the residue ids around the gap by a cutoff
        if start < cutoff and end < (seq_len - 2*cutoff):
            # Case of missing N terminal and the cutoff ends before the 
            # last residue of the chain
            res_ids = list(range(end, end+2*cutoff))
        elif end > seq_len - cutoff and start > 2*cutoff:
            # Case of missing C terminal and the cutoff ends before the
            # first residue of the chain
            res_ids = list(range(start-2*cutoff))
        elif start > cutoff and end < (seq_len - cutoff):
            # The case of an actual gap in the chain and
            # the cutoff ends before the first and last residue of the chain
            res_ids = list(range(start-cutoff, start))+list(range(end, end+cutoff))
        else:
            return None
        return res_ids
    
    def set_around_gap(
            self, ref_chain, mov_chain,
            ref_gap,
            mov_gap=None,
            cutoff=10
        ):

        """
        Set atoms for superposition around a gap/missing residues on the ref 
        chain. Start and end residue index needed. Alignment will be performed on
        backbone atoms. If mov_gap is not specified, identical sequence will be 
        assumed on both chains, and the reference chain gap locations will be applied
        on the move chain 
        """
        ## TODO: add options for on_atoms, and find common res list first 
        ## then run the for-loop on those.

        if not (
            isinstance(ref_chain, PolymerChain) and isinstance(mov_chain, PolymerChain)
        ):
            raise NotImplementedError('PolymerChain class is required to use this method')
        self.ref_chain = ref_chain
        self.mov_chain = mov_chain
        ref_len = len(ref_chain.can_seq)
        mov_len = len(mov_chain.can_seq)
        seq_len = min(ref_len, mov_len)

        ref_res_ids = self.find_valid_residue_range(ref_gap, seq_len, cutoff)
        if mov_gap is None:
            # Assume identical canonical sequence
            mov_res_ids = ref_res_ids
        else:
            mov_res_ids = self.find_valid_residue_range(mov_gap, seq_len, cutoff)

        if ref_res_ids is None:
            warnings.warn('The cutoff value extends out of the available '
                    'residues in the chain. All backbone atoms on both chains '
                    'are selected for superposition')
            ref_common, mov_common = self.get_matching_res()
            ref_atoms, mov_atoms = \
                    self._find_common_backbone_atoms(ref_common, mov_common) 
            self.set_atoms(ref_atoms, mov_atoms)
            return
   
        ref_atoms = []
        mov_atoms = []
        for i, j in zip(mov_res_ids, ref_res_ids):
            if not (i in mov_chain and j in ref_chain):
                continue
            if mov_chain[i].resname != ref_chain[j].resname:
                continue
            for atom in ['N','CA','C','O']:
                if atom not in mov_chain[i]:
                    continue
                ref_atoms.append(ref_chain[j][atom])
                mov_atoms.append(mov_chain[i][atom])
        
        self.set_atoms(ref_atoms, mov_atoms)
    
    def apply_transform(self, entity: PolymerChain=None):
        if entity is None:
            entity = self.mov_chain
        self.apply(entity.get_atoms())

    def show(self):
        show_nglview_multiple([self.ref_chain, self.mov_chain])