import warnings
from typing import List, Tuple, Dict
from Bio.Seq import Seq
from Bio.PDB import PPBuilder
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.PDB.Chain import Chain as _Chain
from Bio.PDB.PDBExceptions import PDBConstructionException
import crimm.StructEntities as cEntities

class BaseChain(_Chain):
    """Base class derived from and Biopython chain object and compatible with
    Biopython's functions"""
    chain_type = "Base Chain"
    def __init__(self, chain_id):
        super().__init__(chain_id)
        # the following attributes are used for topology definitions
        # instantiated but not initiated
        self.undefined_res = None
        self.topo_definitions = None
        self.topo_elements = None
        self.pdbx_description = None

    @property
    def residues(self):
        """Alias for child_list. Returns the list of residues in this chain."""
        return self.child_list

    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent.get_top_parent()

    def reset_atom_serial_numbers(self, include_alt = True):
        """Reset all atom serial numbers in the encompassing entity (the parent
        structure and/or model, if they exist) starting from 1."""
        top_parent = self.get_top_parent()
        if top_parent is not self:
            top_parent.reset_atom_serial_numbers(include_alt=include_alt)
            return
        # no parent, reset the serial number for the entity itself
        i = 1
        for atom in self.get_atoms(include_alt=include_alt):
            atom.set_serial_number(i)
            i+=1

    ## TODO: move this to the structure builder
    def _disordered_reset_residue(self, disordered_residue):
        for resname, child_residue in disordered_residue.child_dict.items():
            if child_residue.id == disordered_residue.id:
                disordered_residue.disordered_select(resname)

    def _disordered_reset_atom(self, residue):
        for atom in residue.get_unpacked_list():
            if atom.is_disordered() == 2:
                first_alt = sorted(atom.child_dict)[0]
                atom.disordered_select(first_alt)

    def reset_disordered_residues(self):
        """Reset the selected child of all disordered residues to the first
        residue (alt loc A) supplied by PDB."""
        for res in self:
            if isinstance(res, cEntities.DisorderedResidue):
                self._disordered_reset_residue(res)
            elif res.disordered == 1:
                self._disordered_reset_atom(res)

    def __repr__(self):
        """Return the chain identifier."""
        return f"<{self.chain_type} id={self.get_id()} Residues={len(self)}>"
        
    def expanded_view(self):
        """Print the expanded view of the chain."""
        repr_str = repr(self)
        if (descr := getattr(self, 'pdbx_description', None)) is not None:
            repr_str += f"\n  Description: {descr}"
        return repr_str
    
    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from crimm.Visualization import show_nglview
        from IPython.display import display
        display(show_nglview(self))
        print(self.expanded_view())

    def get_atoms(self, include_alt = False):
        """Return a generator of all atoms from this chain. If include_alt is True, the 
        disordered residues will be expanded and altloc of disordered atoms will be included."""
        if include_alt:
            all_res = self.get_unpacked_list()
        else:
            all_res = self.child_list
        for res in all_res:
            yield from res.get_atoms(include_alt=include_alt)

    def is_continuous(self):
        """Not implemented in BaseChain. Implementation varies depending on the
        child class"""
        raise NotImplementedError

class Chain(BaseChain):
    """
    General/Unspecified Chain object based on Biopython Chain
    """
    chain_type = 'Chain'
    ## TODO: Implement hetflag check for florescence proteins (chromophore residues)
    def __init__(self, chain_id: str):
        super().__init__(chain_id)
        self._ppb = PPBuilder()
        # There is no way to distinguish the type of the chain. Hence, the
        # nucleic letter codes are limited to one letter codes
        self.letter_3to1_dict = {
            **protein_letters_3to1_extended,
            **{v:v for k, v in nucleic_letters_3to1_extended.items()}
        }
        self.het_res = []
        self.het_resseq_lookup = {}

    def add(self, residue):
        """Add a child to the Entity. Overwrite the Biopython Chain.add method"""
        entity_id = residue.get_id()
        hetflag, resseq, icode = entity_id
        if self.has_id(entity_id):
            raise PDBConstructionException(f"{entity_id} defined twice")
        residue.set_parent(self)
        self.child_list.append(residue)
        self.child_dict[entity_id] = residue
        if hetflag.startswith('H_'):
            self.het_res.append(residue) 
            self.het_resseq_lookup[resseq] = entity_id
        if icode != ' ':
            self.het_resseq_lookup[resseq] = entity_id
    
    def _translate_id(self, id):
        """Translate sequence identifier to tuple form (PRIVATE).

        A residue id is normally a tuple (hetero flag, sequence identifier,
        insertion code). The _translate_id method first looks up if there is a 
        heterogen residue with the resseq id, if not, it translates the 
        sequence identifier to the (" ", sequence identifier, " ") tuple.

        Arguments:
         - id - int, residue resseq

        """
        if id in self.het_resseq_lookup:
            return self.het_resseq_lookup[id]
        if isinstance(id, int):
            id = (" ", id, " ")
        return id
    
    def get_disordered_res(self):
        disordered_res = []
        for res in self:
            if isinstance(res, cEntities.DisorderedResidue):
                disordered_res.append(res)
        return disordered_res
    
    def get_segments(self):
        """
        Build polypeptide segments based on C-N distance criterion
        """
        # This will detect the gap in chain better than using residue sequence
        # numbering
        return self._ppb.build_peptides(self, aa_only=False)

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
        """
        Extract sequence from the residues that are present in chain, regardless
        of any missing atom or even empty residues.
        """
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
        self.seq_lookup: Dict[int, Tuple[str, int, str]] = {}
        if len(self.reported_res) != len(self.can_seq):
            warnings.warn(
                "Total number of reported residues do not match with the "
                "length of reported canonical sequence."
            )
        if self.chain_type == 'Polypeptide(L)':
            self.letter_3to1_dict = protein_letters_3to1_extended
        else:
            self.letter_3to1_dict = {
                **{v:v for k, v in nucleic_letters_3to1_extended.items()},
                **nucleic_letters_3to1_extended
            }

    def __contains__(self, id):
        return super().__contains__(id) or id in self.het_resseq_lookup

    def __getitem__(self, id):
        # Translate the sequence id to the tuple form. We rely on the one-to-one
        # correspendence here
        if id in self.het_resseq_lookup:
            id = self.het_resseq_lookup[id]
        return super().__getitem__(id)
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
            if isinstance(res, cEntities.DisorderedResidue):
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
        if len(self.missing_res) == 0:
            missing_res_ids = []
        else:
            missing_res_ids = list(zip(*self.missing_res))[0]
        return MaskedSeq(missing_res_ids, self.can_seq, self.reported_res[0][0])

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
    
    def sort_residues(self):
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

    def truncate_missing_terminal(self):
        """Remove the missing residues in reported_res list and the sequence info"""
        bg, end = 1, len(self.can_seq)
        trunc_bg = 0
        trunc_end = -1
        terminal_gaps = []
        for gap in self.gaps:
            if bg in gap:
                terminal_gaps.append(gap)
                trunc_bg = len(gap)
            elif end in gap:
                terminal_gaps.append(gap)
                trunc_end = -len(gap)
        if not terminal_gaps:
            return
        self.reported_res = sorted(self.reported_res)[trunc_bg:trunc_end]
        self.can_seq = self.can_seq[trunc_bg:trunc_end]
        known_seq = self.known_seq
        self.known_seq = known_seq.rstrip(known_seq[trunc_end:]).lstrip(known_seq[:trunc_bg])

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
    """A chain of heterogens."""
    chain_type = 'Heterogens'
    def update(self):
        """Update the pdbx_description if only one heterogen exists.
        The description will be assigned directly to that molecule"""
        if len(self) != 1 or self.pdbx_description is None:
            return
        self.child_list[0].pdbx_description = self.pdbx_description

    def __repr__(self):
        """Return the chain identifier."""
        return f"<{self.chain_type} id={self.get_id()} Molecules={len(self)}>"
            
class Macrolide(BaseChain):
    chain_type = 'Macrolide'
    
class Oligosaccharide(BaseChain):
    chain_type = 'Oligosaccharide'

class Solvent(BaseChain):
    chain_type = 'Solvent'

class CoSolvent(BaseChain):
    chain_type = 'CoSolvent'

class Ion(BaseChain):
    chain_type = 'Ion'

class Glycosylation(BaseChain):
    chain_type = 'Glycosylation'

class NucleosidePhosphate(BaseChain):
    chain_type = 'NucleosidePhosphate'

class Ligand(BaseChain):
    chain_type = 'Ligand'

class MaskedSeq(Seq):
    """
    A sequence masked with '-' for any missing residues for visualization purposes.
    The masked sequence is constructed from a PolymerChain class where missing 
    residues are reported. The sequence that is missing will be printed in red 
    if the show() method is called.
    """
    RED = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self, missing_res_ids, can_seq, seq_start, length=None):
        self.color_coded_seq = ''
        masked_seq = ''
        for i, code in enumerate(can_seq, start=seq_start):
            if i in missing_res_ids:
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
    """Convert a Biopython Chain class to general Chain"""
    pchain = Chain(chain.id)
    pchain.set_parent(chain.parent)
    for res in chain:
        res = res.copy()
        pchain.add(res)
    chain.detach_parent()
    return pchain