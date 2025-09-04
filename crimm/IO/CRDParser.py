"""
Module containing the parser class for constructing structures from coord CRD 
files from CHARMM output.

Read a list of atoms from a CHARMM CARD coordinate file (CRD_)
to build a basic biopython/crimm structure.  Reads atom ids (ATOMNO), 
atom names (TYPES), resids (RESID), residue numbers (RESNO), 
residue names (RESNames), segment ids (SEGID) and tempfactor (Weighting).  
Atom element and mass are determined by a lookup table derived from CHARMM36 
and CGENFF residue topology files (rtf).

Residues are detected through a change in resid or resnum, 
while segments are detected according to changes in segid. chains are based on 
segid. The chain ids are assigned based on the alphabet.

"""
import warnings
from string import ascii_uppercase
from collections import namedtuple
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from crimm.IO.StructureBuilder import StructureBuilder
from crimm.IO.PDBParser import protein_letters_3to1, nucleic_letters_3to1
from crimm.Data.element_dict import all_element_dict
from crimm.IO.PDBParser import convert_chains

crd_entry = namedtuple(
    'crd_entry',
    [
        'serial', 'resnum', 'resname', 'atomname', 
        'coord', 'segid', 'resid', 'tempFactor'
    ]
)

class CRDParser:
    """Parse a CHARMM CARD coordinate file for structure coord information.

    Reads the following Attributes:
     - Atomids
     - Atomnames
     - Tempfactors
     - Resids
     - Resnames
     - Resnums
     - Segids

    Determines the following Attributes:
     - Atomtypes
     - Masses
    """
    def __init__(self, include_solvent = True, QUIET = False) -> None:
        self._structure_builder = StructureBuilder()
        self.QUIET = QUIET
        self.include_solvent = include_solvent

    @staticmethod
    def determine_chain_id(i, cur_letters=''):
        cur_letters = ascii_uppercase[i % 26] + cur_letters
        if (j := i//26) > 0:
            return CRDParser.determine_chain_id(j-1, cur_letters)
        return cur_letters

    def get_structure(self, filepath, structure_id = None):
        """Return the structure.

        Arguments:
         :structure_id: string, the id that will be used for the structure
         :filepath: path to mmCIF file, OR an open text mode file handle

        """
        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings(
                    "ignore", category=PDBConstructionWarning
                )
            entries = self.create_namedtuples(filepath)
            if structure_id is None:
                structure_id = filepath.split('/')[-1].split('.')[0]
            self._build_structure(structure_id, entries)

        return self._structure_builder.get_structure()

    def create_namedtuples(self, filepath):
        """Create a list of namedtuples from the CRD file."""
        with open(filepath, 'r') as f:
            lines = [l.rstrip() for l in f.readlines() if not l.startswith('*')]
        entries = []
        for l in lines[1:]:
            entry = l.split()
            (
                serial, resnum, resname, atomname, 
                x, y, z, segid, resid, tempFactor
            ) = entry
            coord = np.array([float(x), float(y), float(z)])
            if resname == 'ILE' and atomname == 'CD':
                atomname = 'CD1'
            cur_entry = crd_entry(
                int(serial), int(resnum), resname, atomname, 
                coord, segid, int(resid), float(tempFactor)
            )
            entries.append(cur_entry)
        return entries
    
    def _build_structure(self, structure_id, entries):
        sb = self._structure_builder
        sb.init_structure(structure_id)
        # Only one model per structure for crd files
        sb.init_model(1)
        # temporarily set to empty dict
        # TODO: Implement a header parser
        sb.header = {} 
        cur_segid = None
        cur_resid = None
        cur_resnum = None
        chain_id_dict = {}
        for entry in entries:
            if entry.segid != cur_segid:
                cur_segid = entry.segid
                sb.init_seg(cur_segid)
                chain_id = self.determine_chain_id(len(chain_id_dict))
                chain_id_dict[chain_id] = cur_segid
                sb.init_chain(chain_id)
            if entry.resid != cur_resid or entry.resnum != cur_resnum:
                cur_resid = entry.resid
                cur_resnum = entry.resnum
                if entry.resname not in protein_letters_3to1 and entry.resname not in nucleic_letters_3to1:
                    field = 'H'
                elif entry.resname in ('HOH', 'WAT', 'TIP3', 'TIP4'):
                    field = 'W'
                else:
                    field = ' '
                sb.init_residue(entry.resname, field, cur_resid, ' ')
            atom_name = entry.atomname
            if atom_name in all_element_dict:
                element = all_element_dict.get(atom_name)
            else:
                element = atom_name[0]
                warnings.warn(
                    f'Element type cannot be determined from atom name {atom_name}! '
                    f'Element is assigned as \'{element}\'.'
                )
                
            sb.init_atom(
                atom_name, entry.coord, entry.tempFactor,
                occupancy = 1.0,
                altloc = ' ',
                fullname = atom_name,
                serial_number = entry.serial,
                element = element
            )
        stucture = sb.structure
        for model in stucture:
            new_chains = convert_chains(model.child_list)
            if not self.include_solvent:
                new_chains = [c for c in new_chains if c.chain_type != 'Solvent']
            for chain in new_chains:
                chain.set_parent(model)
            model.child_list = new_chains
            model.child_dict = {c.id: c for c in new_chains}

        