from string import ascii_uppercase
import warnings
import pathlib
from Bio.PDB.PDBParser import PDBParser as _PDBParser
from Bio.Data.PDBData import (
    protein_letters_3to1, protein_letters_3to1_extended,
    nucleic_letters_3to1, nucleic_letters_3to1_extended
)
from crimm.IO.StructureBuilder import StructureBuilder
from crimm.StructEntities.Chain import Solvent, Heterogens, PolymerChain, Chain

protein_letters_3to1.update({'HSD': 'H', 'HSE': 'H', 'HSP': 'H'})
def check_chain_type(residues, resname_lookup, extended_lookup=None):
    for res in residues:
        if res.resname not in resname_lookup:
            # Check if the residue is a heterogen and if it is in the extended
            # lookup table, heterogen flag will be changed to 'H_{resname}'
            if extended_lookup and (res.resname in extended_lookup):
                hetflag, resseq, icode = res.get_id()
                hetflag = f'H_{res.resname}'
                res.id = (hetflag, resseq, icode)
                warnings.warn(f"Heterogen flag for {res.get_id()} corrected.")
                continue
            return False
    return True

def separate_solvent(residues):
    het = []
    water = []
    for res in residues:
        if res.resname == 'HOH':
            water.append(res)
        else:
            het.append(res)
    return het, water

def find_chain_type(chain):
    residues = []
    het_residues = []
    for res in chain:
        hetflag, resseq, icode = res.get_id()
        if hetflag == " ":
            residues.append(res)
        else:
            het_residues.append(res)
    if residues:
        if check_chain_type(
            residues, protein_letters_3to1, protein_letters_3to1_extended
        ):
            return [('Polypeptide(L)', chain.id, residues)]
        elif check_chain_type(
            residues, nucleic_letters_3to1, nucleic_letters_3to1_extended
        ):
            return [('Polyribonucleotide', chain.id, residues)]
        else:
            warnings.warn(
                f'Chain type cannot be determined for {chain.get_full_id()}'
            )
            return [('Chain', chain.id, residues)]
    else:
        het, water = separate_solvent(het_residues)
        return [('Heterogens', chain.id, het), ('Solvent', chain.id, water)]

def convert_chains(chains):
    organized_residues = []
    for chain in chains:
        organized_residues.extend(find_chain_type(chain))

    chain_sort_dict = {
        'Polypeptide(L)': 0,
        'Polyribonucleotide': 1,
        'Chain': 2, # 'Chain' is a catch-all for 'unknown
        'Heterogens': 3,
        'Solvent': 4
    }

    new_chains = []
    for i, (chain_type, auth_chain_id, residues) in enumerate(
        sorted(organized_residues, key=lambda x: chain_sort_dict[x[0]])
    ):
        if len(residues) == 0:
            continue
        new_chain_id = ascii_uppercase[i]
        if chain_type in ('Polypeptide(L)', 'Polyribonucleotide'):
            new_chain = PolymerChain(
                chain_id = new_chain_id,
                entity_id = None,
                author_chain_id = auth_chain_id,
                chain_type = chain_type,
                known_sequence = '',
                canon_sequence = '',
                reported_res = [],
                reported_missing_res = []
            )
        elif chain_type == 'Chain':
            new_chain = Chain(
                chain_id = new_chain_id,
            )
        elif chain_type == 'Heterogens':
            new_chain = Heterogens(new_chain_id)
        else:
            new_chain = Solvent(new_chain_id)
        for res in residues:
            res.detach_parent()
            if res.resname == 'ILE' and 'CD' in res:
                # ILE could have a CD atom (CHARMM convention) 
                # that is not in the standard PDB format
                atom = res['CD']
                res.detach_child('CD')
                atom.name = 'CD1'
                atom.id = 'CD1'
                atom.fullname = ' CD1'
                res.add(atom)
            new_chain.add(res)
        new_chains.append(new_chain)
    return new_chains

class PDBParser(_PDBParser):
    """PDBParser that returns a Structure with determined chain types."""
    def __init__(
            self, 
            first_model_only = True,
            include_solvent = True,
            strict_parser=True, 
            get_header=False, 
            QUIET=False
        ):
        structure_builder = StructureBuilder()
        self.first_model_only = first_model_only
        self.include_solvent = include_solvent
        PERMISSIVE = not strict_parser
        super().__init__(
            PERMISSIVE, get_header, structure_builder, QUIET, is_pqr=False
        )

    def _get_structure(self, filepath, structure_id=None):
        """Return the structure contained in file."""
        if structure_id is None:
            structure_id = pathlib.Path(filepath).stem
        structure = super().get_structure(structure_id, filepath)
        if self.first_model_only:
            structure.child_list = structure.child_list[:1]
            structure.child_dict = {c.id: c for c in structure.child_list}
        for model in structure:
            new_chains = convert_chains(model.child_list)
            if not self.include_solvent:
                new_chains = [c for c in new_chains if c.chain_type != 'Solvent']
            for chain in new_chains:
                chain.set_parent(model)
            model.child_list = new_chains
            model.child_dict = {c.id: c for c in new_chains}
        return structure

    def get_structure(self, filepath, structure_id=None):
        """Return the structure contained in file."""
        if self.QUIET:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._get_structure(filepath, structure_id)
        return self._get_structure(filepath, structure_id)
    