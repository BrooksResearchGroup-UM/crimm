from Bio.Seq import Seq, MutableSeq
from Bio.PDB import PDBIO, PPBuilder
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.Chain import Chain as _Chain
from Bio.PDB.PDBExceptions import PDBIOException
from Residue import DisorderedResidue, Residue
import warnings
from ChainExceptions import ChainConstructionException, ChainConstructionWarning
from NGLVisualization import load_nglview
from typing import List, Tuple

## TODO: Get rid of chain_type attr. Use isinstance() instead
class BaseChain(_Chain):
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self.chain_type = "Base Chain"

    def reset_atom_serial_numbers(self):
        i = 1
        for atom in self.get_unpacked_atoms():
            atom.serial_number = i
            i+=1

    def _disordered_reset(self, disordered_residue):
        for resname, child_residue in disordered_residue.child_dict.items():
            if child_residue.id == disordered_residue.id:
                disordered_residue.disordered_select(resname)

    def reset_disordered_residues(self):
        for res in self:
            if isinstance(res, DisorderedResidue):
                self._disordered_reset(res)

    def __repr__(self):
        """Return the peptide chain identifier."""
        repr_str = f"<{self.chain_type} id={self.get_id()} Residues/Molecules={len(self)}>"
        return repr_str

    def _repr_html_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from IPython.display import display
        view = load_nglview(self)
        display(view)

    def get_unpacked_atoms(self):
        atoms = []
        for res in self.get_unpacked_list():
            atoms.extend(res.get_unpacked_atoms())
        return atoms
    
    def get_pdb_str(self, reset_serial = True, include_alt = True):
        """
        Get the PDB format string for all atoms in the chain
        """
        # This is copied and modified from Bio.PDB.PDBIO
        if reset_serial:
            self.reset_atom_serial_numbers()
        io = PDBIO()
        pdb_string = ''
        if include_alt:
            get_child = lambda x: x.get_unpacked_list()
        else:
            get_child = lambda x: x.child_list

        for residue in get_child(self):
            hetfield, resseq, icode = residue.id
            resname = residue.resname
            segid = residue.segid

            for atom in get_child(residue):
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
        pdb_string += 'TER\n'
        return pdb_string

class Chain(BaseChain):
    """
    General/Unspecified Chain object based on Biopython Chain
    """
    ## TODO: Implement hetflag check for florecence proteins (chromophore residues)
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self.chain_type = 'Chain'
        self._ppb = PPBuilder()
        # There is no way to distinguish the type of the chain. Hence, the
        # nucleic letter codes are limited to one letter codes
        self.letter_3to1_dict = {
            **protein_letters_3to1_extended,
            **{v:v for k, v in nucleic_letters_3to1_extended.items()}
        }
    
    ## FIXME: move this to Polymer Chain that is constructed from mmCIF
    ## Since mmCIF sourced structure has unique resseq and no duplicate exists,
    ## but PDB parser can generate duplicated resseq, and this method will fail.
    def find_het_by_seq(self, resseq):
        modified_het_ids = []
        for res in self:
            if resseq == res.id[1]:
                modified_het_ids.append(res.id)
        return modified_het_ids
    
    @staticmethod
    def is_res_modified(res):
        if not isinstance(res, DisorderedResidue):
            return res.id[0].startswith('H_')

        for child_res in res.child_dict.values():
            if child_res.id[0].startswith('H_'):
                return True
        
    
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
            if res.resname not in self.letter_3to1_dict:
                one_letter_code = 'X'
            else:
                one_letter_code = self.letter_3to1_dict[res.resname]
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
            if res.resname not in self.letter_3to1_dict:
                one_letter_code = 'X'
            else:
                one_letter_code = self.letter_3to1_dict[res.resname]
            seq+=one_letter_code
        return seq
    
    def is_continuous(self):
        """
        Check if the chain is continuous (no gaps). Missing segments on 
        the C and N terminals will be ignored.
        """
        segments = self.get_segments()
        # If the strucutures C-N distance are all within the criterion,
        # it is continuous
        return len(segments) == 1

class PolymerChain(Chain):
    """
    A derived Chain class for holding only polypeptide or nucleotide residues with 
    sequence id reset to be continuous and start from 1 (MMCIF standard). 
    Author reported canonical sequence and missing residues are required for init. 
    """
    def __init__(
        self,
        chain_id: str,
        entity_id: int,
        author_chain_id: str,
        chain_type: str,
        known_sequence: str,
        canon_sequence: str,
        reported_res: List[Tuple[int, str]],
        reported_missing_res: List[Tuple[int, str]] = None
    ):
        """
        Attributes:
            instance_attribute (str): The instance attribute
        """
        if chain_id is None:
            chain_id = entity_id
        super().__init__(chain_id)
        self.entity_id = entity_id
        self.author_chain_id = author_chain_id
        self.chain_type = chain_type[0].upper()+chain_type[1:]
        self.reported_res = reported_res
        if reported_missing_res is None:
            reported_missing_res = []
        self.reported_missing_res = reported_missing_res
        self.known_seq = Seq(known_sequence)
        self.can_seq = Seq(canon_sequence)
        self.seq = Seq('')
        self.cur_missing_res = reported_res
        self.masked_seq = Seq('-'*len(canon_sequence))
        self.gaps = None
        if len(self.reported_res) != len(self.can_seq):
            warnings.warn(
                "Total number of reported residues do not match with the "
                "length of reported canonical sequence.",
                ChainConstructionWarning
            )
        if self.chain_type == 'Polypeptide(L)':
            self.letter_3to1_dict = protein_letters_3to1_extended
        else:
            self.letter_3to1_dict = {
                **{v:v for k, v in nucleic_letters_3to1_extended.items()},
                **nucleic_letters_3to1_extended
            }

    def update(self):
        """Update the following attribute based on the current present residues:
        seq, missing_res, masked_seq, gaps
        """
        self.seq = self.extract_present_seq()
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
        # If not, we also check the sequence here, because after copying gap residues, C-N 
        # distance might be very large before structure minimizations.
        sequence_segments = [seq for seq in self.masked_seq.split('-') if len(seq) > 0]
        return len(sequence_segments) == 1
    
class Heterogens(BaseChain):
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self.chain_type = 'Heterogens'

class Macrolide(BaseChain):
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self.chain_type = 'Macrolide'
class Oligosaccharide(BaseChain):
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self.chain_type = 'Oligosaccharide'

class Solvent(BaseChain):
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self.chain_type = 'Solvent'

def convert_chain(chain: _Chain):
    """Convert a Biopython Chain class to CHARMM Chain"""
    pchain = Chain(chain.id)
    pchain.set_parent(chain.parent)
    for res in chain:
        res = res.copy()
        pchain.add(res)
    chain.detach_parent()
    return pchain