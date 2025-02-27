from string import ascii_uppercase
from crimm.StructEntities.Chain import Solvent, Ion
from copy import copy
import numpy as np

def index_to_letters(index, letter = ''):
    """Enumerate a sequence of letters based on the alphabet from a integer number.
    The length of the sequence corresponds to the exponent of 26 from 
    the given index, and the last letter is determined by (index % 26).
    e.g. 
        index_to_letters(0) = 'A'
        index_to_letters(25) = 'Z'
        index_to_letters(26) = 'AA'
        index_to_letters(701) = 'ZZ'
        index_to_letters(702) = 'AAA'
    """
    if index < 26:
        letter = ascii_uppercase[index] + letter
        return letter

    letter = ascii_uppercase[index%26] + letter
    index = index // 26 - 1
    return index_to_letters(index, letter)

def letters_to_index(letters):
    """For a given string enumerated from alphabet, return the associated index.
    This is the inverse function of `index_to_letters`
    e.g.
        letters_to_index('A') = 0
        letters_to_index('Z') = 25
        letters_to_index('AA') = 26
        letters_to_index('ZZ') = 701
        letters_to_index('AAA') = 702
    """
    index = -1
    for i, l in enumerate(reversed(letters)):
        cur_index = ascii_uppercase.find(l)+1
        index += cur_index * (26 ** i)
    return index

def get_coords(entity, include_alt=False):
    """Get atom coordinates from any structure entity level or a list of entities.
    If a list of entity is provided, the coordinates will be an Nx3 array, where 
    N is the total number of atoms in the list"""
    if isinstance(entity, list):
        return np.concatenate([get_coords(child) for child in entity])
    if not hasattr(entity, 'level'):
        raise TypeError(
            'get_coord takes a Biopython/crimm structure object, '
            f'while {type(entity)} is provided')
    if entity.level == 'A':
        return entity.coord
    return np.array([a.coord for a in entity.get_atoms(include_alt)])

def _rename_chains_by_order(chains):
    for i, chain in enumerate(chains):
        chain.id = index_to_letters(i)

def rename_chains_by_order(entity):
    """Rename the chain IDs based on the order listed in the parent model so that
    the IDs start from 'A' and are continuous in the alphabetical order."""
    if entity.level not in ('S', 'M'):
        raise ValueError(
            "Only Structure or Model level entities accepted! "
            f"While {entity.level} level entity {entity} is supplied. "
        )

    if entity.level == 'S':
        for model in entity:
            _rename_chains_by_order(model.child_list)

    else:
        _rename_chains_by_order(entity.child_list)

def combine_hetero_chains(
        chains, new_chain_id, enforce_same_type=True, reset_resid=True
    ):
    """Combine a list of chains into a single chain with a specified chain ID."""
    parent = chains[0].parent
    # get an empty chain but retain the chain type of the first chain
    new_chain = copy(chains[0])
    new_chain.child_list = []
    new_chain.child_dict = {}
    new_chain.detach_parent()
    new_chain.id = new_chain_id
    all_residues = []
    all_description = []
    for chain in chains:
        if chain.chain_type in (
            'Polypeptide(L)', 'Polyribonuleotide', 'Polydeoxyribonucleotide'
        ):
            raise ValueError(
                f"Chain {chain.id} is a biopolymer chain and cannot be combined with "
                "other chains using this function."
            )
        if enforce_same_type and chain.chain_type != new_chain.chain_type:
            raise ValueError(
                f"Chain {chain.id} has a different chain type {chain.chain_type} "
                f"from the first chain {new_chain.id} with chain type {new_chain.chain_type}."
            )
        parent.detach_child(chain.id)
        all_residues.extend(chain.residues)

    for i, residue in enumerate(all_residues, start=1):
        residue.detach_parent()
        if reset_resid:
            het_flag, resseq, icode = residue.id
            residue.id = (het_flag, i, icode)
        if hasattr(residue, 'pdbx_description') and residue.pdbx_description not in all_description:
            all_description.append(residue.pdbx_description)
        new_chain.add(residue)
    new_chain.pdbx_description = ', '.join(all_description)

    return new_chain

def get_charges(chain):
    """Get the total charge of a generated chain."""
    total_charge = 0
    for atom in chain.get_atoms():
        total_charge += atom.topo_definition.charge
    return round(total_charge, 3)
