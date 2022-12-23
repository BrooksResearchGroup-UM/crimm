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

def convert_chain(chain: Chain):
    pchain = PeptideChain(chain.id)
    pchain.set_parent(chain.parent)
    for res in chain:
        res = res.copy()
        hetflag, resseq, icode = res.get_id()
        if hetflag != " ": 
        # This will not be needed as a filter, since any ligand or
        # solvent will be filtered out by not having the "label_seq_id" in
        # mmCIF entries
        # When constructing the chain with author_residue=False, those not on 
        # the polypeptide chain will be ignored
            if not pchain._is_res_modified(hetflag):
                continue
            pchain.modified_res.append(res)
        if res.is_disordered():
            pchain._add_disordered_res(res)
            continue
        pchain.add(res)
    chain.detach_parent()
    return pchain

class PeptideChain(Chain):
    """
    PeptideChain object based on Biopython Chain
    """
    ## TODO: Implement hetflag check for florecence proteins (chromophore residues)
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self._ppb = PPBuilder()
        
    def find_het_by_seq(self, resseq):
        modified_het_ids = []
        for res in self:
            if resseq == res.id[1]:
                modified_het_ids.append(res.id)
        return modified_het_ids
    
    @staticmethod
    def is_res_modified(res):
        if not isinstance(res, DisorderedResidue):
            for child_res in res.child_dict.values():
                if child_res.id[0].startswith('H_'):
                    return True
        return res.id[0].startswith('H_')
    
    def get_modified_res(self):
        modified_res = []
        for res in self:
            if self.is_res_modified(res):
                modified_res.append(res)
        return modified_res
    
    def get_disordered_res(self):
        disordered_res = []
        for res in self:
            if isinstance(res, DisorderedResidue):
                disordered_res.append(res)
        return disordered_res
    
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
        super().__init__(chain.id)
        self.reported_res = reported_res
        self.reported_missing_res = reported_missing_res
        self.known_seq = Seq(known_sequence)
        self.can_seq = Seq(canon_sequence)
        self.seq = Seq('')
        self.cur_missing_res = reported_res
        self.masked_seq = Seq('-'*len(canon_sequence))
        if len(self.reported_res) != len(self.can_seq):
            warnings.warn(
                "Total number of reported residues do not match with the "
                "length of reported canonical sequence.",
                ChainConstructionWarning
            )
        
    def update(self):
        """Update the following attribute based on the current present residues:
        seq, missing_res, masked_seq, gaps
        """
        self.seq = self.extract_present_seq()
        self.cur_missing_res = self.find_missing_res()
        if set(self.cur_missing_res) != set(self.reported_missing_res):
            warnings.warn(
                "Reported missing residues do not match with the current "
                "missing residues from the chain.",
                ChainConstructionWarning
            )
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