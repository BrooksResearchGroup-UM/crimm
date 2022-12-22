from Bio.Seq import Seq, MutableSeq
from Bio.PDB import PDBIO, PPBuilder
from Bio.Align import PairwiseAligner
from Bio.PDB.Polypeptide import protein_letters_3to1_extended, Polypeptide
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBExceptions import PDBIOException
from Bio.PDB.Residue import DisorderedResidue, Residue
import warnings
from ChainExceptions import ChainConstructionException, ChainConstructionWarning
from typing import List, Tuple
   
class PeptideChain(Chain):
    """
    PeptideChain object based on Biopython Chain
    """
    ## TODO: Implement hetflag check for florecence proteins (chromophore residues)
    def __init__(self, chain: Chain):
        id = chain.get_id()
        super().__init__(id)
        self.set_parent(chain.parent)
        self._ppb = PPBuilder()
        self.modified_res = []
        self.disordered_res = dict()
        for res in chain:
            res = res.copy()
            hetflag, resseq, icode = res.get_id()
            if hetflag != " ": 
            # This will not be needed as a filter, since any ligand or
            # solvent will be filtered out by not having the "label_seq_id" in
            # mmCIF entries
            # When constructing the chain with author_residue=False, those not on 
            # the polypeptide chain will be ignored
                if not self._is_res_modified(hetflag):
                    continue
                self.modified_res.append(res)
            if res.is_disordered():
                self._add_disordered_res(res)
                continue
            self.add(res)
        chain.detach_parent()

    def _add_disordered_res(self, res):
        resseq = res.id[1]
        if resseq not in self.disordered_res:
            # if the disordered residue has not been recorded
        #     disordered = DisorderedResidue(res.id)
        #     self.disordered_res[resseq] = disordered
        #     self.add(disordered)
        # self.disordered_res[resseq].disordered_add(res)
            # We only add the first one onto the chain
            self.add(res)
            self.disordered_res[resseq] = []
        self.disordered_res[resseq].append(res)
        
    def _is_res_modified(self, hetflag: str):
        resname = hetflag.lstrip("H_")
        if resname in ('CRO','GYS','PIA','NRQ'):
            # These are known peptide-derived chromophores that
            # do not have a one-letter-code representation, but
            # they are indeed modified residues
            # This is not a complete list, and many other chromophore
            # residue will slip through!
            return True
        return resname in protein_letters_3to1_extended

    def get_segments(self):
        """
        Build polypeptide segments based on C-N distance criterion
        """
        # This will detect the gap in chain better than using residue sequence
        # numbering
        return self._ppb.build_peptides(self, aa_only=False)

    def extract_all_seq(self):
        """
        Extract sequence from all residues in chain. Even if the residues is empty
        and contains no atom, it will be included in the sequence.
        """
        seq = Seq('')
        for res in sorted(self, key = lambda x: x.id[1:]):
            if res.resname not in protein_letters_3to1_extended:
                one_letter_code = 'X'
            else:
                one_letter_code = protein_letters_3to1_extended[res.resname]
            seq+=one_letter_code
        return seq

    def extract_segment_seq(self):
        """
        Extract sequence from the residues that are present in chain. Residues that 
        have missing C and N backbone atoms are skipped.
        """
        seq = Seq('')
        for segment in self.get_segments():
            seq += segment.get_sequence()
        return seq
    
    def extract_present_seq(self):
        seq = Seq('')
        for res in sorted(self, key = lambda x: x.id[1:]):
            if len(res.child_list) == 0:
                continue
            if res.resname not in protein_letters_3to1_extended:
                one_letter_code = 'X'
            else:
                one_letter_code = protein_letters_3to1_extended[res.resname]
            seq+=one_letter_code
        return seq
    
    def reset_atom_serial_numbers(self):
        i = 1
        for res in self:
            for atom in res:
                atom.serial_number = i
                i+=1

    def __repr__(self):
        """Return the peptide chain identifier."""
        return f"<Peptide Chain id={self.get_id()}>"

    def _repr_html_(self):
        """Return the nglview interactive visualization window"""
        from IPython.display import display

        try:
            import nglview as nv
        except ImportError:
            warnings.warn(
                "WARNING: nglview not found! Install nglview to show\
                protein structures. \
                http://nglviewer.org/nglview/latest/index.html#installation"
            )
            return self.__repr__()

        view = nv.NGLWidget()
        self.load_nglview(view)
        display(view)

    def load_nglview(self, view):
        """
        Load pdb string into nglview instance
        """
        blob = self.get_pdb_str()
        ngl_args = [{'type': 'blob', 'data': blob, 'binary': False}]
        view._ngl_component_names.append(self.get_id())
        view._remote_call("loadFile",
                          target='Stage',
                          args=ngl_args,
                          kwargs= {'ext':'pdb',
                          'defaultRepresentation':True}
                          )

    def get_pdb_str(self):
        """
        Get the PDB format string for all atoms in the chain
        """
        # This is copied and modified from Bio.PDB.PDBIO
        self.reset_atom_serial_numbers()
        io = PDBIO()
        pdb_string = ''
        for residue in self.get_unpacked_list():
            hetfield, resseq, icode = residue.id
            resname = residue.resname
            segid = residue.segid
            resid = residue.id[1]
            if resid > 9999:
                e = f"Residue number ('{resid}') exceeds PDB format limit."
                raise PDBIOException(e)

            for atom in residue.get_unpacked_list():
                atom_number = atom.serial_number
                s = io._get_atom_line(
                    atom,
                    hetfield,
                    segid,
                    atom_number,
                    resname,
                    resseq,
                    icode,
                    self.get_id(),
                )
                pdb_string += s
        return pdb_string
    
    def is_continuous(self):
        """
        Check if the chain is continuous (no gaps). Missing segments on 
        the C and N terminals will be ignored.
        """
        segments = self.get_segments()
        # If the strucutures C-N distance are all within the criterion,
        # it is continuous
        return len(segments) == 1
            

class MaskedPeptideChain(PeptideChain):
    """
    A derived Chain class for holding only polypeptide residues with sequence id reset to be
    continuous and start from 1 (Missing terminal residues will be included). 
    Author reported canonical sequence and missing residues are required to init. 
    All missing residues will be and represented by an empty residue container (no atom exists).
    """
    def __init__(self, chain: Chain, 
        canon_sequence: str, 
        reported_missing_res: List[Residue]  = []):
        """ 
        Attributes:
            instance_attribute (str): The instance attribute
        """

        self.is_seq_id_reset = False
        self.reported_missing_res = reported_missing_res
        self.can_seq = Seq(canon_sequence)
        super().__init__(chain)
        # Add in the missing residues (they are empty container of residue instances)
        for res in reported_missing_res:
            if res.id in self:
                warnings.warn("Reported missing residue {} is present in chain"\
                        .format(res.id))
                continue
            self.add(res)
        # set the order of residues based on the sequence number
        self.child_list = sorted(self.child_list, key = lambda x: x.id[1:])
        
        # This uses PPBuild to determine segments based on C-N distance 
        # for matching with canonical sequence. This only needs to be done during init
        self._reset_res_ids()
        self.seq = self.extract_all_seq()
        self.update()
    
    def update(self):
        """Update the following attribute based on the current present residues:
        seq, missing_res, masked_seq, gaps
        """
        self.missing_res = self.find_missing_res()
        self.masked_seq = self.create_missing_seq_mask(self.missing_res)
        self.gaps = self.find_gaps(self.missing_res)

    def _reset_res_ids(self):
        for i in range(len(self)):
            res = self.child_list[i]
            hetflag =  res.id[0]
            # set a temp flag on the insertion code to avoid clash 
            # in dict keys
            res.id = (hetflag, i, 'temp')

        for i in range(len(self)):
            res = self.child_list[i]
            hetflag = res.id[0]
            # remove temp icode and add offset, res seq id is 1-indexed
            new_id = i+1
            res.id = (hetflag, new_id, ' ')

        self.is_seq_id_reset = True

    def align_with_can_seq(self):
        """
        Perform sequence alignment to determine the gap in the current residue sequence.
        The scoring function maximally penalize mismatch and gap opening while ignore gap extensions
        to encourage preservation of continuous segments of polypeptide.
        """
        aligner = PairwiseAligner()
        aligner.match_score = 10
        aligner.mismatch_score = -1000
        aligner.target_open_gap_score = -10000
        aligner.query_open_gap_score = -100
        aligner.query_extend_gap_score = 0
        all_seq = self.extract_all_seq()
        alignments = aligner.align(self.can_seq, all_seq)
        can_seq_range, seq_range = alignments[0].aligned
        return can_seq_range, seq_range
    
    # TODO: change to property and setter
    def find_missing_res(self):
        """
        Find missing residue in the chain.
        Current sequence will be compared to the reported canonical sequence to 
        determine the missing residue (any missing C and N termini will be detected). 
        """
        
        missing_res = []
        for res in self:
            _, resseq, _ = res.get_id()
            if len(res.child_list) == 0:
                missing_res.append((resseq, res.resname))
            
        return missing_res
    
    def _is_subsequence(self, seq1, seq2):
        return (seq1 in seq2) or (seq2 in seq1)
    
    def create_missing_seq_mask(self, missing_res):
        """
        Create a sequence masked with '-' for any residue that is missing C and N
        backbone atoms
        """
        cur_seq = MutableSeq(self.extract_present_seq()) 
        start_id = 1 # res seq id is 1-indexed
        
        for id, res_code in missing_res:
            cur_seq.insert(id - start_id, '-')
        if len(cur_seq) != len(self.can_seq):
            warnings.warn('Canonical sequence and masked sequence mismatch!')
            if not self._is_subsequence(cur_seq, self.can_seq) \
                    and len(self.get_segments()) > 1:
                # This means that the author did not report the missing residue 
                # correctly, and we are gonna have trouble here
                raise IndexError('Missing residue not reported, but the reported '
                    'canonical sequence indicates that there are missing residues.')
        return Seq(cur_seq)

    def find_gaps(self, missing_res):
        """
        Group gap residues into sets for comparison purposes
        """
        gaps = []
        if len(missing_res) == 0:
            return gaps
        
        prev_id = missing_res[0][0]
        cur_set = set()
        for id, res_code in missing_res[1:]:
            cur_set.add(prev_id)
            if id - prev_id > 1:
                gaps.append(cur_set)
                cur_set = set()
            prev_id = id
        cur_set.add(prev_id)
        gaps.append(cur_set)
        return gaps

    def is_continuous(self):
        """
        Check if the chain is continuous (no gaps). Missing segments on 
        the C and N terminals will be ignored.
        """
        segments = self.get_segments()
        if len(segments) == 1:
            # If the strucuture's C-N distances are all within the criterion,
            # it is continuous
            return True
        self.update()
        # If not, we also check the sequence here, because C-N 
        # distance might be very large before structure minimizations.
        sequence_segments = [seq for seq in self.masked_seq.split('-') if len(seq) > 0]
        return len(sequence_segments) == 1


class CanonicalPeptideChain(PeptideChain):
    """
    A derived Chain class for holding only polypeptide residues and reset sequence residue ids
    to match author provided canonical sequence. Author reported canonical sequence from PDB
    is required to init this class. Any missing residues will NOT be in this class.
    This method is slightly inferior that the MaskedPeptideChain classin finding missing residue 
    in tricky scenarios (very messy PDB data).
    """
    def __init__(
            self, chain: Chain, 
            canon_sequence: str
        ):
        """ 
        Attributes:
            instance_attribute (str): The instance attribute
        """
        self.is_seq_id_reset = False
        self.can_seq = Seq(canon_sequence)
        super().__init__(chain)
        # This uses PPBuild to determine segments based on C-N distance 
        # for matching with canonical sequence. This only needs to be done during init
        self.correct_res_ids()
        self.update()
        
    def update(self):
        """Update the following attribute based on the current present residues:
        seq, missing_res, masked_seq, gaps
        """
        self.seq = self.extract_segment_seq()
        self.missing_res = self.find_missing_res()
        self.masked_seq = self.create_missing_seq_mask(self.missing_res)
        self.gaps = self.find_gaps(self.missing_res)
    
    def correct_res_ids(self):
        """
        Update the residue sequence ids to match the reported canonical sequence id
        Return True if the ids are successfully sync'd with canonical sequence
        """
        segments = self.get_segments()
        cur_seq = self.extract_segment_seq()
        if cur_seq in self.can_seq and len(segments) == 1:
            # If current sequence is already continuous and a 
            # substring/subsequence of the canonical sequence
            # `offset` is res seq id (1-indexed)
            offset = self.can_seq.find(self.seq) + 1
            if offset != 0:
                # Reset all residue index by the same offset
                self._reset_res_ids(offset)
            return True
        
        start_ids = self._find_seg_start_ids(segments)
        if start_ids == None:
            # This is usually due to the peptide builder sliced the segment
            # so small that the recursive find method would return multiple
            # locations of the matches
            # Sequence alignment will be performed to join the segments
            # and return the correct starting idx for each segment
            segments, start_ids = self._adjust_segments_from_alignment()
            if len(segments) == 0:
                return False
            
        self._offset_segment_res_ids(segments, start_ids)
        return True

    def _reset_res_ids(self, offset):
        for i in range(len(self)):
            res = self.child_list[i]
            # set a temp flag on the insertion code to avoid clash 
            # in dict keys
            res.id = (' ', i, 'temp')

        for i in range(len(self)):
            res = self.child_list[i]
            # remove temp icode and add offset
            new_id = i+offset
            res.id = (' ', new_id, ' ')

        self.is_seq_id_reset = True

    def _find_seg_start_ids(self, segments):
        """
        Compare current residues with canonical sequence for the correct
        residue starting index.
        """
        start_ids = []
        cur_start = 0
        for segment in segments:
            sequence = segment.get_sequence()
            all_locs = self._find_all_locations(self.can_seq, sequence, cur_start)

            if len(all_locs) == 0:
                # If not found or more than one location present,
                # We need to fall back to sequence alignment based matching
                warnings.warn('Sequence segment not found in canonical sequence!'
                    '\nSequence alignment based method will be used to determine '
                    'missing residues.'
                    '\nSegment:\n{}\nCanonical sequence\n{}'\
                    .format(sequence, self.can_seq))
                return None
            elif len(all_locs) > 1:
                warnings.warn('Multiple locations found in canonical sequence!'
                    '\nSequence alignment based method will be used to determine '
                    'missing residues.'
                    '\nSegment:\n{}\nCanonical sequence\n{}'\
                    .format(sequence, self.can_seq))
                return None
            
            loc = all_locs[0]
            # The residue sequence number is 1-indexed
            start_ids.append(loc+1)
            cur_start = loc+len(sequence)
        return start_ids
    
    def _find_all_locations(self, seq, seg, start = 0):
        """
        Recursively find all locations of a substring pattern from the parent
        string
        """
        loc = seq.find(seg, start)
        if loc > -1:
            return [loc] + self._find_all_locations(seq, seg, loc+1)
        return []
    
    def _offset_segment_res_ids(self, segments, correct_starts):
        """
        Update the residue sequence ids to match canonical sequence id
        """
        reset = False
        for segment, start_id in zip(segments, correct_starts):
            # Correct residue sequence number at the start of the segment
            # (1-indexed)
            correct_seq_id = start_id
            for res in segment:
                # set a temp flag on the insertion code to avoid clash 
                # in dict keys
                if res.id[1] != correct_seq_id:
                    reset = True
                    res.id = (' ', correct_seq_id, 'temp')
                    correct_seq_id += 1

        if reset:
            for res in self.get_residues():
                _, correct_seq_id, _ = res.get_id()
                # remove the "temp" insertion code
                res.id = (' ', correct_seq_id, ' ')
        self.is_seq_id_reset = reset

    def _adjust_segments_from_alignment(self):
        """
        Regroup the residues into segments of polypeptide to match the result of sequence alignment.
        This is used as the last resort when the recursive find function for segment matching fails.
        """
        auto_segments = (seg for seg in self.get_segments())
        adjusted_segments = []
        start_ids = []
        can_seq_range, _ = self._align_with_can_seq()
        for st, end in can_seq_range:
            can_segment_seq = self.can_seq[st:end]
            cur_segment = Polypeptide()
            while len(cur_segment) < len(can_segment_seq):
                try:
                    cur_segment.extend(next(auto_segments))
                except StopIteration:
                    warnings.warn('Canonical sequence and present sequence mismatch!\n')
                    return [], []
                
            if len(cur_segment) != len(can_segment_seq):
                warnings.warn('Canonical sequence and present sequence mismatch!\n'
                    'Sequence Segment:\n{}\nCanonical Sequence Segment:\n{}\n'\
                    .format(cur_segment.get_sequence(), can_segment_seq))
                return [], []
            
            start_ids.append(st+1)
            adjusted_segments.append(cur_segment)
        return adjusted_segments, start_ids    
    
    def _align_with_can_seq(self):
        """
        Perform sequence alignment to determine the gap in the current residue sequence.
        The scoring function maximally penalize mismatch and gap opening while ignore gap extensions
        to encourage preservation of continuous segments of polypeptide.
        """
        aligner = PairwiseAligner()
        aligner.match_score = 10
        aligner.mismatch_score = -1000
        aligner.target_open_gap_score = -10000
        aligner.query_open_gap_score = -100
        aligner.query_extend_gap_score = 0
        alignments = aligner.align(self.can_seq, self.seq)
        can_seq_range, seq_range = alignments[0].aligned
        return can_seq_range, seq_range
    
    def find_missing_res(self):
        """
        Find missing residue in the chain.
        Current sequence will be compared to the reported canonical sequence to 
        determine the missing residue (any missing C and N termini will be detected). 
        """
        
        res_set = set()
        for res in self.get_residues():
            _, resseq, _ = res.get_id()
            resi_name = res.resname
            if resi_name in ('UNK', 'ACE', 'NH2'):
                # These are commonly marked as X in sequence
                one_letter_code = 'X'
            else:
                one_letter_code = protein_letters_3to1_extended[resi_name]
            res_set.add((resseq, one_letter_code))

        start, end = 1, len(self.can_seq)+1
        # create a list of continuous ids from canonical sequence for comparison
        all_res_ids = range(start, end)
        # zip it with residues' one letter codes
        all_res_set = set(zip(all_res_ids, self.can_seq))
        missing_res_set = all_res_set.difference(res_set)
        missing_res = []
        for idx, one_letter_code in sorted(list(missing_res_set)):
            if one_letter_code == 'X':
                missing_res.append((idx, 'UNK'))
            else:
                missing_res.append((idx, protein_letters_1to3[one_letter_code]))
            
        return missing_res
    
    def create_missing_seq_mask(self, missing_res):
        """
        Create a sequence masked with '-' for missing residues
        """
        cur_seq = MutableSeq(self.extract_segment_seq()) 
        start_id = 1
        
        for id, res_code in missing_res:
            cur_seq.insert(id - start_id, '-')
        if len(cur_seq) != len(self.can_seq):
            if (self.can_seq in cur_seq):
                warnings.warn('Reported canonical sequence is not complete, more'
                    'residues present in the current chain than the reported sequence.')
            else:
                raise ValueError('Canonical sequence and present residue index '
                    'mismatch! Masked sequence:\n{}\n'
                    'Canonical sequence:\n{}'.format(cur_seq, self.can_seq))
        return Seq(cur_seq)

    def find_gaps(self, missing_res):
        """
        Group gap residues into sets for comparison purposes
        """
        gaps = []
        if len(missing_res) == 0:
            return gaps
        
        prev_id = missing_res[0][0]
        cur_set = set()
        for id, res_code in missing_res[1:]:
            cur_set.add(prev_id)
            if id - prev_id > 1:
                gaps.append(cur_set)
                cur_set = set()
            prev_id = id
        cur_set.add(prev_id)
        gaps.append(cur_set)
        return gaps

    def is_continuous(self):
        """
        Check if the chain is continuous (no gaps). Missing segments on 
        the C and N terminals will be ignored.
        """
        segments = self.get_segments()
        if len(segments) == 1:
            # If the strucuture's C-N distances are all within the criterion,
            # it is continuous
            return True
        self.update()
        # If not, we also check the sequence here, because C-N 
        # distance might be very large before structure minimizations.
        sequence_segments = [seq for seq in self.masked_seq.split('-') if len(seq) > 0]
        return len(sequence_segments) == 1


class StandardChain(PeptideChain):
    """
    A derived Chain class for holding only polypeptide residues with sequence id reset to be
    continuous and start from 1 (Missing terminal residues will be included). 
    Author reported canonical sequence and missing residues are required for init. 
    """
    def __init__(self, chain: Chain,
        known_sequence: str,
        canon_sequence: str,
        reported_res: List[Tuple[int, str]],
        reported_missing_res: List[Tuple[int, str]] = []):
        """
        Attributes:
            instance_attribute (str): The instance attribute
        """
        self.reported_res = reported_res
        self.reported_missing_res = reported_missing_res
        self.known_seq = Seq(known_sequence)
        self.can_seq = Seq(canon_sequence)
        if len(self.reported_res) != len(self.can_seq):
            warnings.warn(
                "Total number of reported residues do not match with the "
                "length of reported canonical sequence.",
                ChainConstructionWarning
            )
        super().__init__(chain)
        
        self.seq = self.extract_present_seq()
        self.cur_missing_res = self.find_missing_res()
        
        if set(self.cur_missing_res) != set(reported_missing_res):
            warnings.warn(
                "Reported missing residues do not match with the current "
                "missing residues from the chain.",
                ChainConstructionWarning
            )

        self.update()
    
    def update(self):
        """Update the following attribute based on the current present residues:
        seq, missing_res, masked_seq, gaps
        """
        self.cur_missing_res = self.find_missing_res()
        self.masked_seq = self.create_missing_seq_mask(self.cur_missing_res)
        self.gaps = self.find_gaps(self.cur_missing_res)
        self.child_list = sorted(self.child_list, key=lambda x: x.id[1:])
    
    # TODO: change to property and setter
    def find_missing_res(self):
        """
        Find missing residue in the chain.
        Current sequence will be compared to the reported canonical sequence to 
        determine the missing residue (any missing C and N termini will be detected). 
        """
        present_res = set()
        for res in self:
            if isinstance(res,DisorderedResidue):
                # Record both residues as present from the disordered residue
                for resname, child_res in res.child_dict.items():
                    _, resseq, _ = child_res.get_id()
                    present_res.add((resseq, resname))
            else:
                _, resseq, _ = res.get_id()
                present_res.add((resseq, res.resname))

        missing_res = []
        for i, reported in self.reported_res:
            if (i, reported) not in present_res:
                missing_res.append((i, reported))
        
        return missing_res
    
    def _check_with_can_seq(self, masked_seq):
        if len(masked_seq) != len(self.can_seq):
            warnings.warn(
                    'Canonical sequence and masked sequence mismatch!'
            )
        for masked_char, char in zip(masked_seq, self.can_seq):
            if masked_char == char or masked_char == '-' or char == 'X':
                continue
            warnings.warn(
                    'Canonical sequence and masked sequence mismatch!'
            )
        
    def create_missing_seq_mask(self, missing_res):
        """
        Create a sequence masked with '-' for any residue that is missing C and N
        backbone atoms
        """
        cur_seq = MutableSeq(self.extract_present_seq()) 
        start_id = 1 # res seq number is 1-indexed
        
        for id, _ in missing_res:
            cur_seq.insert(id - start_id, '-')
        self._check_with_can_seq(cur_seq)

        return Seq(cur_seq)

    def find_gaps(self, missing_res):
        """
        Group gap residues into sets for comparison purposes
        """
        gaps = []
        if len(missing_res) == 0:
            return gaps
        
        prev_id = missing_res[0][0]
        cur_set = set()
        for id, res_name in missing_res[1:]:
            cur_set.add(prev_id)
            if id - prev_id > 1:
                gaps.append(cur_set)
                cur_set = set()
            prev_id = id
        cur_set.add(prev_id)
        gaps.append(cur_set)
        return gaps

    def is_continuous(self):
        """
        Check if the chain is continuous (no gaps). Missing segments on 
        the C and N terminals will be ignored.
        """
        segments = self.get_segments()
        if len(segments) == 1:
            # If the strucuture's C-N distances are all within the criterion,
            # it is continuous
            return True
        self.update()
        # If not, we also check the sequence here, because C-N 
        # distance might be very large before structure minimizations.
        sequence_segments = [seq for seq in self.masked_seq.split('-') if len(seq) > 0]
        return len(sequence_segments) == 1