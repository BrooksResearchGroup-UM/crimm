import warnings
from typing import List, Tuple
from Bio.Seq import Seq, MutableSeq
from Bio.PDB import PDBIO, PPBuilder
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.PDB.Chain import Chain as _Chain
from Residue import DisorderedResidue
from ChainExceptions import ChainConstructionWarning
from NGLVisualization import load_nglview

## TODO: Get rid of chain_type attr. Use isinstance() instead
class BaseChain(_Chain):
    """Base class for Biopython and CHARMM compatible object with nglview visualizations"""
    chain_type = "Base Chain"
    def __init__(self, chain_id):
        super().__init__(chain_id)
        # the following attributes are used for topology definitions
        # instantiated but not initiated
        self.undefined_res = None
        self.topo_definitions = None
        self.pdbx_description = None

    def reset_atom_serial_numbers(self):
        """Reset all atom serial numbers starting from 1."""
        i = 1
        for atom in self.get_unpacked_atoms():
            atom.serial_number = i
            i+=1

    def _disordered_reset(self, disordered_residue):
        for resname, child_residue in disordered_residue.child_dict.items():
            if child_residue.id == disordered_residue.id:
                disordered_residue.disordered_select(resname)

    def reset_disordered_residues(self):
        """Reset the selected child of all disordered residues to the first
        residue (alt loc A) supplied by PDB."""
        for res in self:
            if isinstance(res, DisorderedResidue):
                self._disordered_reset(res)

    def __repr__(self):
        """Return the peptide chain identifier."""
        repr_str = f"<{self.chain_type} id={self.get_id()} Residues/Molecules={len(self)}>"
        if (descr := getattr(self, 'pdbx_description', None)) is not None:
            repr_str += f"\n  Description: {descr}"
        return repr_str

    def _repr_html_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from IPython.display import display
        view = load_nglview(self)
        display(view)
        return

    def get_unpacked_atoms(self):
        atoms = []
        for res in self.get_unpacked_list():
            atoms.extend(res.get_unpacked_atoms())
        return atoms
    
    @staticmethod
    def _get_child(parent, include_alt):
        if include_alt:
            return parent.get_unpacked_list()
        return parent.child_list
    
    def get_pdb_str(self, reset_serial = True, include_alt = True):
        """
        Get the PDB format string for all atoms in the chain
        """
        # This is copied and modified from Bio.PDB.PDBIO
        if reset_serial:
            self.reset_atom_serial_numbers()
        io = PDBIO()
        pdb_string = ''
        # Since chain_ids are from label_asym_id in mmCIF, and 
        # for larger structures with mmCIF entity naming scheme,
        # it would ran out of the alphabet and start using two letter codes.
        # But here, we are forcing the one character chain_id in order to 
        # comply with PDB file spec.
        chain_id = self.get_id()[0]
        for residue in self._get_child(self, include_alt):
            hetfield, resseq, icode = residue.id
            resname = residue.resname
            segid = residue.segid

            for atom in self._get_child(residue, include_alt):
                atom_number = atom.serial_number
                atom_line = io._get_atom_line(
                    atom,
                    hetfield,
                    segid,
                    atom_number,
                    resname,
                    resseq,
                    icode,
                    chain_id,
                )
                pdb_string += atom_line
        pdb_string += 'TER\n'
        return pdb_string

    def load_topo_definition(self, topology_definitions: dict):
        """Load topology definition for all residues from a dictionary of ResidueDefinition
        objects. Any residue that does not have a corresponding definition in the dictionary
        will be placed in `self.undefined_res`."""
        self.topo_definitions = topology_definitions
        self.undefined_res = []
        for residue in self:
            if (
                residue.resname not in topology_definitions
            ) and (
                residue.topo_definition is None
            ):
                warnings.warn(
                    f'No topology definition for {residue.resname}!'
                )
                self.undefined_res.append(residue)
                continue

            if residue.topo_definition is not None:
                warnings.warn(
                    f"Overwriting residue topology definition: {residue}"
                )
            residue.load_topo_definition(topology_definitions[residue.resname])

    def is_continuous(self):
        """Not implemented in BaseChain. Implementation varies depending on the
        child class"""
        raise NotImplementedError

class Chain(BaseChain):
    """
    General/Unspecified Chain object based on Biopython Chain
    """
    chain_type = 'Chain'
    ## TODO: Implement hetflag check for florecence proteins (chromophore residues)
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self._ppb = PPBuilder()
        # There is no way to distinguish the type of the chain. Hence, the
        # nucleic letter codes are limited to one letter codes
        self.letter_3to1_dict = {
            **protein_letters_3to1_extended,
            **{v:v for k, v in nucleic_letters_3to1_extended.items()}
        }

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
    Author reported canonical sequence and missing residues are required for its
    construction.
    This is the default class for polypeptide and polynucleotide parsed from mmCIF.
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

    @property
    def seq(self):
        """
        Get the current sequence from the present residues only
        """
        return self.extract_present_seq()
    
    @property
    def missing_res(self):
        """
        Get the current missing residues in the chain.
        Currently present residues will be compared to the the list of author- 
        reported residues to determine the missing ones.
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

    @property
    def masked_seq(self):
        """
        Get a sequence masked with '-' for any residue that is missing CA and N
        backbone atoms
        """
        missing_res_ids = list(zip(*self.missing_res))[0]
        return MaskedSeq(missing_res_ids, self.can_seq)
    
    @property
    def gaps(self):
        """
        Group gap residues into sets for comparison purposes
        """
        missing_res = self.missing_res
        gaps = []
        if len(missing_res) == 0:
            return gaps
        
        prev_idx = missing_res[0][0]
        cur_set = set()
        for idx, res_name in missing_res[1:]:
            cur_set.add(prev_idx)
            if idx - prev_idx > 1:
                gaps.append(cur_set)
                cur_set = set()
            prev_idx = idx
        cur_set.add(prev_idx)
        gaps.append(cur_set)
        return gaps
    
    def update(self):
        """Update the ordering of the residues in child list by resseq
        """
        self.child_list = sorted(self.child_list, key=lambda x: x.id[1:])
    
    def find_het_by_resseq(self, resseq):
        """Return a list of heterogens for a residue seq id."""
        modified_het_ids = []
        for res in self:
            if resseq == res.id[1]:
                modified_het_ids.append(res.id)
        return modified_het_ids

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

        # If not, we also check the sequence here, because after copying gap residues, C-N 
        # distance might be very large before structure minimizations.
        sequence_segments = [
            seq for seq in self.masked_seq.split('-') if len(seq) > 0
        ]
        return len(sequence_segments) == 1

class Heterogens(BaseChain):
    chain_type = 'Heterogens'
    def update(self):
        """Update the pdbx_description if only one heterogen exists.
        The description will be assigned directly to that molecule"""
        if len(self) != 1 or self.pdbx_description is None:
            return
        self.child_list[0].pdbx_description = self.pdbx_description

    def to_rdkit_mols(self):
        """Convert all molecules in the heterogen chain to rdkit mols"""
        return [res.to_rdkit() for res in self]
            
class Macrolide(BaseChain):
    chain_type = 'Macrolide'
    
class Oligosaccharide(BaseChain):
    chain_type = 'Oligosaccharide'

class Solvent(BaseChain):
    chain_type = 'Solvent'

class MaskedSeq(Seq):
    """
    A sequence masked with '-' for any missing residues. The masked sequence
    is constructed from a PolymerChain class where missing residues are 
    reported. The sequence that is missing will be printed in red if the show() 
    method is called.
    """
    RED = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self, missing_res_ids, can_seq, length=None):
        self.color_coded_seq = ''
        masked_seq = ''
        for i, code in enumerate(can_seq):
            if i+1 in missing_res_ids:
                self.color_coded_seq += f"{self.RED}{code}{self.ENDC}"
                masked_seq += '-'
            else:
                self.color_coded_seq += code
                masked_seq += code
        super().__init__(masked_seq, length)
        self.is_matched = self.check_with_can_seq(can_seq)

    def check_with_can_seq(self, can_seq):
        """Check if the MaskedSequence matches with the canonical sequence for the 
        present residues. Unidentified residues on canonical sequence will be 
        skipped for the check."""
        msg = 'Canonical sequence and masked sequence mismatch!'
        if len(self) != len(can_seq):
            warnings.warn(msg)
            return False
        for masked_char, char in zip(self, can_seq):
            if masked_char != char and masked_char != '-' and char != 'X':
                warnings.warn(msg)
                return False
        return True

    def show(self):
        """Display the entire sequence where the missing residues are colored
        in red."""
        print(self.color_coded_seq)


def convert_chain(chain: _Chain):
    """Convert a Biopython Chain class to whaler Chain"""
    pchain = Chain(chain.id)
    pchain.set_parent(chain.parent)
    for res in chain:
        res = res.copy()
        pchain.add(res)
    chain.detach_parent()
    return pchain