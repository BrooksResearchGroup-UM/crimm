import os
import warnings
import pickle
import subprocess
from typing import List, Tuple
from copy import deepcopy, copy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_1to3

from crimm import StructEntities as Entities
from crimm.IO.PRMParser import categorize_lines, parse_line_dict
from crimm.IO.RTFParser import RTFParser
from crimm.Modeller import ResidueFixer
from crimm.Modeller.TopoFixer import fix_chain
from crimm.Data.cgenff_mass_dict import CGENFF_MASS_TABLE

from crimm.Adaptors.RDKitConverter import RDKitHetConverter, MolToMol2Block

toppar_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../Data/toppar')
)
rtf_path_dict = {
    "protein": os.path.join(toppar_dir, 'prot.rtf'),
    "nucleic": os.path.join(toppar_dir, 'na.rtf'),
    "lipid": os.path.join(toppar_dir, 'lipid.rtf'),
    "carb": os.path.join(toppar_dir, 'carb.rtf'),
    "ethers": os.path.join(toppar_dir, 'ethers.rtf'),
    "cgenff": os.path.join(toppar_dir, 'cgenff.rtf'),
    "water_ions": os.path.join(toppar_dir, 'water_ions.rtf'),
}

prm_path_dict = {
    "protein": os.path.join(toppar_dir, 'prot.prm'),
    "nucleic": os.path.join(toppar_dir, 'na.prm'),
    "lipid": os.path.join(toppar_dir, 'lipid.prm'),
    "carb": os.path.join(toppar_dir, 'carb.prm'),
    "ethers": os.path.join(toppar_dir, 'ethers.prm'),
    "cgenff": os.path.join(toppar_dir, 'cgenff.prm'),
    "water_ions": os.path.join(toppar_dir, 'water_ions.prm'),
}

chain_type_def_lookup = {
    "Polypeptide(L)": "protein",
    "Polyribonucleotide": "nucleic",
    "Polynucleotide": "nucleic",
    "Solvent": "water_ions",
    "Ion": "water_ions"
}

protein_n_term_patch_correction = {
    'PRO': 'PROP',
    'GLY': 'GLYP'
}

def _find_atom_in_residue(
        residue: Entities.Residue, atom_name: str
    )->Entities.Atom:
    if atom_name.startswith('+') or atom_name.startswith('-'):
        return None
    if atom_name in residue:
        return residue[atom_name]
    if atom_name.startswith('H'):
        return residue.missing_hydrogens[atom_name]
    return residue.missing_atoms[atom_name]

def _find_atom_from_neighbor(
        cur_residue: Entities.Residue, atom_name: str
    )->Entities.Atom:
    """Get atom from neighbor residue. (Private function)"""
    resseq = cur_residue.id[1]
    chain = cur_residue.parent
    if atom_name.startswith('-') and resseq-1 in chain:
        atom_name = atom_name[1:]
        neighbor_residue = chain[resseq-1]
        return _find_atom_in_residue(neighbor_residue, atom_name)
    elif atom_name.startswith('+') and resseq+1 in chain:
        atom_name = atom_name[1:]
        neighbor_residue = chain[resseq+1]
        return _find_atom_in_residue(neighbor_residue, atom_name)
    return None

def get_bonds_within_residue(residue: Entities.Residue)->List[Entities.Bond]:
    """Return a list of bonds within the residue (peptide bonds linking neighbor 
    residues are excluded). Raise ValueError if the topology definition is 
    not loaded."""
    if residue.topo_definition is None:
        raise ValueError(
            'Topology definition is not loaded for this residue!'
        )
    bonds = []
    bond_dict = residue.topo_definition.bonds
    for bond_type, bond_list in bond_dict.items():
        for atom_name1, atom_name2 in bond_list:
            atom1 = _find_atom_in_residue(residue, atom_name1)
            atom2 = _find_atom_in_residue(residue, atom_name2)
            if atom1 is None or atom2 is None:
                # one of the atoms has to be in the neighbor residue
                continue
            bonds.append(
                Entities.Bond(atom1, atom2, bond_type)
            )
    return bonds

def atom_add_neighbors(atom1: Entities.Atom, atom2: Entities.Atom):
    """Add atom2 to atom1's neighbors and vice versa"""
    atom1.neighbors.add(atom2)
    atom2.neighbors.add(atom1)

def residue_trace_atom_neigbors(residue: Entities.Residue)->List[Entities.Bond]:
    """Trace all bonds within the residue and add the atoms to each other's
    neighbors list. Return a list of bonds within the residue."""
    bonds = get_bonds_within_residue(residue)
    for bond in bonds:
        a1, a2 = bond
        atom_add_neighbors(a1, a2)
    return bonds

def clear_atom_neighbors(entity):
    """Clear all neighbors of atoms in an entity"""
    if isinstance(entity, Entities.Atom):
        entity.neighbors = set()
        return
    elif not hasattr(entity, 'get_atoms'):
        raise ValueError(
            'Entity does not have get_atoms method, cannot clear neighbors.'
        )
    for atom in entity.get_atoms():
        atom.neighbors = set()

def chain_trace_atom_neighbors(
        chain: Entities.PolymerChain, inter_res_bonding_atoms: Tuple[str],
        bond_type='single'
    )->List[Entities.Bond]:
    """Trace all bonds within the chain and add the atoms to each other's
    neighbors list. Return a list of bonds within the chain."""
    end_atom, start_atom = inter_res_bonding_atoms
    # e.g. this would be C-N for peptide,
    # thus end_atom is C and start_atom is N
    chain.sort_residues()
    # Clearing neighbors is needed for patching and regenerating topology
    clear_atom_neighbors(chain)
    all_bonds = []
    for i, cur_res in enumerate(chain.residues[:-1]):
        next_res = chain.residues[i+1]
        bonds = residue_trace_atom_neigbors(cur_res)
        all_bonds.extend(bonds)
        # add inter-residue neigbors from peptide/nucleotide bond

        a1 = _find_atom_in_residue(cur_res, end_atom)
        a2 = _find_atom_in_residue(next_res, start_atom)
        atom_add_neighbors(a1, a2)
        inter_res_bond = Entities.Bond(a1, a2, bond_type)
        all_bonds.append(inter_res_bond)

    last_res = chain.residues[-1]
    bonds = residue_trace_atom_neigbors(last_res)
    all_bonds.extend(bonds)
    return all_bonds

def _add_dihedral(cur_atom, nei_atom, second_nei_atom, dihedral_set):
    """Add dihedral angle to the dihedral set. (Private function)"""
    for third_nei_atom in second_nei_atom.neighbors:
        if third_nei_atom == nei_atom:
            continue
        dihe = Entities.Dihedral(cur_atom, nei_atom, second_nei_atom, third_nei_atom)
        dihedral_set.add(dihe)

def _add_angle_and_dihedral(cur_atom, nei_atom, angle_set, dihedral_set):
    """Add angle and dihedral angle to the angle and dihedral sets. (Private function)"""
    for second_nei_atom in nei_atom.neighbors:
        if second_nei_atom == cur_atom:
            continue
        angle = Entities.Angle(cur_atom, nei_atom, second_nei_atom)
        angle_set.add(angle)
        _add_dihedral(cur_atom, nei_atom, second_nei_atom, dihedral_set)

def traverse_graph(cur_atom, angle_set, dihedral_set, visited_set):
    """Traverse the graph of atoms and add all angles and dihedral angles to the
    angle and dihedral sets."""
    visited_set.add(cur_atom)

    for nei_atom in cur_atom.neighbors:
        _add_angle_and_dihedral(cur_atom, nei_atom, angle_set, dihedral_set)
        if nei_atom in visited_set:
            continue
        traverse_graph(nei_atom, angle_set, dihedral_set, visited_set)

def _get_improper_from_atom_names(
        residue: Entities.Residue, atom_names: Tuple[str]
    )->Entities.Improper:
    """Get improper from atom names. (Private function)"""
    atoms = []
    for atom_name in atom_names:
        if atom_name.startswith('-') or atom_name.startswith('+'):
            atom = _find_atom_from_neighbor(residue, atom_name)
        else:
            atom = _find_atom_in_residue(residue, atom_name)
        if atom is None:
            return None
        atoms.append(atom)
    a1, a2, a3, a4 = atoms
    for atom in (a2, a3, a4):
        if atom not in a1.neighbors:
            raise ValueError(
                'Improper angle definition is incorrect: '
                f'Atom {atom} is not neighbor of atom {a1}!'
            )
    return Entities.Improper(a1, a2, a3, a4)

def _is_terminal_or_orphan_residue(residue: Entities.Residue)->bool:
    """Check if the residue is a terminal residue or does not belong to a chain. 
    (Private function)"""
    chain: Entities.PolymerChain = residue.parent
    if chain is None:
        # orphan residue
        return True
    return residue in (chain.residues[0], chain.residues[-1])

def residue_get_impropers(residue: Entities.Residue)->List[Entities.Improper]:
    """Return a list of improper angles within the residue. Raise ValueError if the 
    topology definition is not loaded."""
    if residue.topo_definition is None:
        raise ValueError(
            'Topology definition is not loaded for this residue!'
        )
    impropers = []
    for impr_atom_names in residue.topo_definition.impropers:
        improper = _get_improper_from_atom_names(residue, impr_atom_names)
        if improper is None:
            if not _is_terminal_or_orphan_residue(residue):
                warnings.warn(
                    f'Cannot find improper {impr_atom_names} in residue {residue}'
                )
            continue
        impropers.append(improper)
    return impropers

def get_impropers(chain: Entities.PolymerChain)->List[Entities.Improper]:
    """Return a list of improper angles within the chain. Raise ValueError if the 
    topology definition is not loaded."""
    impropers = []
    for res in chain.residues:
        impropers.extend(residue_get_impropers(res))
    return impropers

def _get_cmap_from_atom_names(res, cmap_atom_names):
    """Get cmap from atom names. (Private function)"""
    raise NotImplementedError

def get_cmap(chain: Entities.PolymerChain)->List[Entities.CMap]:
    """Return a list of CMap terms within the chain. Raise ValueError if the 
    topology definition is not loaded."""
    cmaps = []
    for res in chain.residues:
        if res.topo_definition is None:
            raise ValueError(
                'Topology definition is not loaded for this residue!'
            )
        for cmap_atom_names in res.topo_definition.cmaps:
            cmap = _get_cmap_from_atom_names(res, cmap_atom_names)
            if cmap is None:
                if not _is_terminal_or_orphan_residue(res):
                    warnings.warn(
                        f'Cannot find cmap {cmap_atom_names} in residue {res}'
                    )
                continue
            cmaps.append(cmap)
    return cmaps

def excute_cgenff(cgenff_path, input_mol2_block):
    """Excute cgenff to generate topology and parameter files for a given molecule.
    Takes mol2 block as input and return the output as string."""
    completed = subprocess.run(
        [cgenff_path],
        input=input_mol2_block,
        encoding='ascii',
        capture_output=True,
        check=False
    )

    if completed.returncode != 0:
        raise ValueError(
            f'CGENFF Error:\n{completed.stderr}\n{completed.stdout}'
        )

    return completed.stdout

class BaseTopology:
    topo_types = [
        'bonds', 'angles', 'dihedrals', 'impropers'
    ]
    def __init__(self):
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None
        self.atom_lookup = None
        self.missing_param_dict = None
        self.containing_entity = None
        self._visited_atoms = None
        self.containing_entity = None

    def __iter__(self):
        for topo_type_name in self.topo_types:
            yield topo_type_name, getattr(self, topo_type_name)

    def __repr__(self) -> str:
        if self.containing_entity is None:
            return "<EmptyTopology>"
        s = f"<Topology of {self.containing_entity} with "
        for attr, value in self:
            if value is None:
                n = 0
            else:
                n = len(value)
            s += f"{attr}={n}, "
        s = s[:-2] + ">"
        return s

    def delete_atom_related_elements(self, atom: Entities.Atom):
        """Delete all topology elements related to the atom"""
        for topo_type_name, topo_list in self:
            if topo_list is None:
                continue
            remove_list = []
            for topo in topo_list:
                if atom in topo:
                    remove_list.append(topo)
            for topo in remove_list:
                topo_list.remove(topo)

    def update(self):
        """Update the topology elements"""
        self.find_topo_elements(self.containing_entity)
        self.create_atom_lookup_table()

    def create_atom_lookup_table(self) -> dict:
        """Create a lookup table for all topology elements for a given atom in the chain"""
        atom_lookup = {
            atom:{
                'bonds': [],
                'angles': [],
                'dihedrals': [],
                'impropers': []
            }
            for atom in self._visited_atoms
        }

        for bond in self.bonds:
            for atom in bond:
                atom_lookup[atom]['bonds'].append(bond)
            
        for angle in self.angles:
            for atom in angle:
                atom_lookup[atom]['angles'].append(angle)

        for dihedral in self.dihedrals:
            for atom in dihedral:
                atom_lookup[atom]['dihedrals'].append(dihedral)

        for improper in self.impropers:
            for atom in improper:
                atom_lookup[atom]['impropers'].append(improper)

        self.atom_lookup = atom_lookup

    def find_topo_elements(self, entity):
        """Find all topology elements in the entity"""
        raise NotImplementedError

class ModelTopology:
    """A class object that stores topology elements (bond, angles, dihe, etc) 
    for a model."""
    topo_types = [
        'bonds', 'angles', 'dihedrals', 'impropers', 'cmap'
    ]
    def __init__(self, model):
        """Find all topology elements from the model"""
        if not isinstance(model, Entities.Model):
            raise ValueError(
                'Model is not an instance of Model!'
                f' {type(model)} is provided.'
            )
        self.disu_bonds = []
        self.containing_entity = model
        self._create_disu_bonds(model)

    def __iter__(self):
        for topo_type_name in self.topo_types:
            yield topo_type_name, getattr(self, topo_type_name)

    def _gather_topo(self, element_name):
        elements = []
        for chain in self.containing_entity:
            if chain.topology is None:
                continue
            if hasattr(chain.topology, element_name):
                chain_elements = getattr(chain.topology, element_name)
                if chain_elements is not None:
                    elements.extend(chain_elements)
        return elements
            
    @property
    def bonds(self):
        return self.disu_bonds+self._gather_topo('bonds')
    @property
    def angles(self):
        return self._gather_topo('angles')
    @property
    def dihedrals(self):
        return self._gather_topo('dihedrals')
    @property
    def impropers(self):
        return self._gather_topo('impropers')
    @property
    def cmap(self):
        return self._gather_topo('cmap')
    # @property
    # def nonbonded(self):
    #     return self._gather_topo('nonbonded')
    
    def __repr__(self):
        repr_str = ''
        if self.containing_entity is None:
            return "<EmptyTopology>"
        s = f"<Topology of {self.containing_entity} with "
        for attr, value in self:    
            if value is None:
                n = 0
            else:
                n = len(value)
            s += f"{attr}={n}, "
        if self.disu_bonds:
            s += f'(disulfide bonds={len(self.disu_bonds)}), '
        s = s[:-2] + ">"
        repr_str += s + '\nTopology by Chains:\n'
        
        for chain in self.containing_entity:
            if chain.topology is None:
                repr_str += f'<Topology of {chain} not generated!>\n'
            else:
                repr_str += repr(chain.topology) + '\n'
        return repr_str
                        
    def _create_disu_bonds(self, model: Entities.Model):
        if 'disulf' in model.connect_atoms:
            patcher = ResiduePatcher()
            for (a1, a2) in model.connect_atoms['disulf']:
                res1 = a1.parent
                res2 = a2.parent
                chain1 = res1.parent
                chain2 = res2.parent
                if chain1.topology is None or chain2.topology is None:
                    warnings.warn(
                        f'Disulfide bond found between {res1} and {res2}! '
                        'But topology not generated for the chains.'
                    )
                    continue
                if 'HG1' in res1:
                    chain1.topology.delete_atom_related_elements(res1['HG1'])
                    res1.detach_child('HG1')
                if 'HG1' in res2:
                    chain2.topology.delete_atom_related_elements(res2['HG1'])
                    res2.detach_child('HG1')
                warnings.warn(
                    f'Disulfide bond found between {res1} and {res2}! '
                    'Removing hydrogen HG1 from the residues.'
                )
                cys_def = patcher.patch_disulfide(
                    res1.topo_definition, res2.topo_definition
                )
                res1.topo_definition = cys_def
                res2.topo_definition = cys_def
                res1['SG'].topo_definition = cys_def['SG']
                res2['SG'].topo_definition = cys_def['SG']
                res1['CB'].topo_definition = cys_def['CB']
                res2['CB'].topo_definition = cys_def['CB']
                chain1.topology.update()
                chain2.topology.update()
                disu_bond = Entities.Bond(a1, a2, 'single')
                self.disu_bonds.append(disu_bond)
    
    
class HeterogenTopology(BaseTopology):
    """A class object that stores topology elements (bond, angles, dihe, etc) 
    for a heterogen residue, e.g. ligand, water, ions, etc."""
    def __init__(self):
        super().__init__()
        
    def load_chain(self, heterogen_chain):
        """Find all topology elements from the heterogen chain"""
        self.containing_entity = heterogen_chain
        self.find_topo_elements(heterogen_chain)
        self.create_atom_lookup_table()
        return self
                        
    def find_topo_elements(self, heterogen_chain: Entities.Chain):
        """Find all topology elements in the chain"""
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self._visited_atoms = []
        for residue in heterogen_chain:
            cur_bonds = residue_trace_atom_neigbors(residue)
            if len(cur_bonds) == 0:
                continue
            self.bonds.extend(cur_bonds)
            visited_atoms = set()
            angles = set()
            dihedrals = set()
            seed_atom = self._find_seeding_atom(residue)
            traverse_graph(seed_atom, angles, dihedrals, visited_atoms)
            self._visited_atoms.extend(list(visited_atoms))
            self.angles.extend(list(angles))
            self.dihedrals.extend(list(dihedrals))
            self.impropers.extend(residue_get_impropers(residue))

    def _find_seeding_atom(self, residue):
        """Find the first atom to start the search"""
        for atom in residue.get_atoms():
            if len(atom.neighbors) > 0:
                return atom
        raise ValueError(
            f'No seeding atom found in residue {residue}!'
            'This is possibly due to the heterogen residue '
            'not being defined in the topology file.'
        )

class ChainTopology(BaseTopology):
    """A class object that stores topology elements (bond, angles, dihe, etc) 
    of bio-polymer chains, e.g. Protein, RNA, DNA."""
    topo_types = [
        'bonds', 'angles', 'dihedrals', 'impropers', 'cmap'
    ]
    def __init__(self):
        super().__init__()
        self.cmap = None

    def load_chain(self, chain):
        """Find all topology elements from the chain"""
        self.containing_entity = chain
        self.find_topo_elements(chain)
        self.create_atom_lookup_table()
        return self

    def _find_seeding_atom(self):
        """Find the first atom to start the search"""
        for atom in self.containing_entity.get_atoms():
            if len(atom.neighbors) > 0:
                return atom

    ## TODO: get Cmap from the topology rtf file
    def find_topo_elements(self, chain: Entities.Chain):
        """Find all topology elements in the chain"""
        if chain.undefined_res:
            raise ValueError(
                "Cannot find topology elements for a chain with undefined residues! "
                f"Undefined residues: {chain.undefined_res}. This is possibly due "
                "to heterogen residues. Use TopologyGenerator.generate() "
                "with coerce=True to coerce the heterogen name into canoncal residue "
                "if necessary."
            )

        if not chain.is_continuous():
            raise ValueError(
                "Cannot generate topology elements for a chain with "
                f"discontinuous backbone! Missing residues: {chain.gaps}. Use "
                "loop modeling tool to construct the missing residues first."
            )

        if chain.chain_type == 'Polypeptide(L)':
            inter_res_bond = ('C','N') # peptide bond
        elif chain.chain_type == 'Polyribonucleotide':
            inter_res_bond = ("O3'",'P') # phosphodiester bond
        else:
            raise NotImplementedError("Chain type not supported!")

        self.bonds = chain_trace_atom_neighbors(chain, inter_res_bond)
        visited_atoms = set()
        angles = set()
        dihedrals = set()
        seed_atom = self._find_seeding_atom()
        traverse_graph(seed_atom, angles, dihedrals, visited_atoms)
        self._visited_atoms = list(visited_atoms)
        self.angles = list(angles)
        self.dihedrals = list(dihedrals)
        self.impropers = get_impropers(chain) 


class ParameterLoader:
    ic_position_dict = {
        'R(I-J)': (0, 1),
        'T(I-J-K)': (0, 1, 2),
        'T(J-K-L)': (1, 2, 3),
        'R(K-L)': (2, 3),
        'T(I-K-J)': (0, 2, 1),
        'R(I-K)': (0, 2),
    }
    """A dictionary that stores parameters for CHARMM force field."""
    def __init__(self, entity_type=None):
        self.param_dict = {}
        self._raw_data_strings = []
        if entity_type is not None:
            self.load_type(entity_type=entity_type)

    def load_type(self, entity_type:str):
        """Load parameters from a CHARMM prm file."""
        entity_type = entity_type.lower()
        if entity_type not in prm_path_dict:
            raise ValueError(f'No parameter file for {entity_type}')
        filename = prm_path_dict[entity_type]
        with open(filename, 'r', encoding='utf-8') as f:

            self._raw_data_strings = [
                l.rstrip() for l in f.readlines() #if not skip_line(l)
            ]
            param_line_dict = categorize_lines(self._raw_data_strings)
        self.param_dict.update(parse_line_dict(param_line_dict))

    def __repr__(self):
        n_bonds = len(self.param_dict['bonds'])
        n_angles = len(self.param_dict['angles'])
        n_urey_bradley = len(self.param_dict['urey_bradley'])
        n_dihedrals = len(self.param_dict['dihedrals'])
        n_impropers = len(self.param_dict['improper'])
        n_cmaps = len(self.param_dict['cmap'])
        n_nonbonds = len(self.param_dict['nonbonded'])
        n_nonbond14s = len(self.param_dict['nonbonded14'])
        n_nbfixes = len(self.param_dict['nbfix'])
        return (
            f'<ParameterDict Bond: {n_bonds}, Angle: {n_angles}, '
            f'Urey Bradley: {n_urey_bradley}, Dihedral: {n_dihedrals}, '
            f'Improper: {n_impropers}, CMAP: {n_cmaps}, '
            f'Nonbond: {n_nonbonds}, Nonbond14: {n_nonbond14s}, '
            f'NBfix: {n_nbfixes}>'
        )

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key):
        return self.param_dict[key]

    def _get_param(self, param_dict: dict, key):
        return (
            param_dict.get(key) or param_dict.get(tuple(reversed(key)))
        )

    def _get_from_choices(self, param_dict: dict, matching_orders: tuple):
        for choice in matching_orders:
            if value := self._get_param(param_dict, choice):
                return value

    def get_bond(self, key):
        """Get bond parameters for a given bond instance."""
        bond_dict = self.param_dict['bonds']
        return self._get_param(bond_dict, key)

    def get_angle(self, key):
        """Get angle parameters for a given angle instance."""
        angle_dict = self.param_dict['angles']
        return self._get_param(angle_dict, key)

    def get_dihedral(self, key):
        """Get dihedral parameters for a given dihedral instance."""
        A, B, C, D = key
        matching_orders = (
            (A, B, C, D),
            ('X', B, C, 'X'),
        )
        dihedral_dict = self.param_dict['dihedrals']
        return self._get_from_choices(dihedral_dict, matching_orders)

    def get_improper(self, key):
        """Get improper parameters for a given improper instance."""
        A, B, C, D = key
        matching_orders = (
            ( A,   B,   C,  D ),
            ( A,  'X', 'X', D ),
            ('X',  B,   C,  D ),
            ('X',  B,   C, 'X'),
            ('X', 'X',  C,  D )
        )
        improper_dict = self.param_dict['improper']
        return self._get_from_choices(improper_dict, matching_orders)

    def get_nonbonded(self, key):
        """Get nonbonded parameters for a given atom type."""
        nonbonded_param = self.param_dict['nonbonded'][key]
        return nonbonded_param

    def get_from_topo_element(self, topo_element):
        """Get the parameter for a given topology element"""
        if isinstance(topo_element, Entities.Bond):
            return self.get_bond(topo_element.atom_types)
        elif isinstance(topo_element, Entities.Angle):
            return self.get_angle(topo_element.atom_types)
        elif isinstance(topo_element, Entities.Dihedral):
            return self.get_dihedral(topo_element.atom_types)
        elif isinstance(topo_element, Entities.Improper):
            return self.get_improper(topo_element.atom_types)
        else:
            raise ValueError('Invalid topology element type')

    def _apply_to_element_list(self, topo_type, topo_element_list):
        """Apply the parameter for a list of topology element"""
        if topo_type == 'bonds':
            param_get_func = self.get_bond
        elif topo_type == 'angles':
            param_get_func = self.get_angle
        elif topo_type == 'dihedrals':
            param_get_func = self.get_dihedral
        elif topo_type == 'impropers':
            param_get_func = self.get_improper
        else:
            raise ValueError('Invalid topology element type')
        no_param_list = []
        for topo_element in topo_element_list:
            param = param_get_func(topo_element.atom_types)
            if param is None:
                no_param_list.append(topo_element)
            else:
                topo_element.param = param
        return no_param_list
    
    def apply(self, topo_element_container: ChainTopology):
        """Apply the parameter for a list of topology element"""
        if not isinstance(topo_element_container, (ChainTopology, HeterogenTopology)):
            raise TypeError(
                'Invalid argument type provided! Topology'
                f' is expected. {type(topo_element_container)} is provided.'
            )
        missing_param_dict = {}
        
        for topo_type, topo_element_list in topo_element_container:
            if topo_element_list is None:
                warnings.warn(
                    f'No {topo_type} found in '
                    f'{topo_element_container.containing_entity}.')
                continue
            no_param_list = self._apply_to_element_list(
                topo_type, topo_element_list
            )
            if no_param_list:
                warnings.warn(
                    f'{len(no_param_list)} {topo_type} failed to find '
                    'parameters.'
                )
                missing_param_dict[topo_type] = no_param_list
        topo_element_container.missing_param_dict = missing_param_dict

    @staticmethod
    def _find_atom_type(atom_name_list, residue_definition):
        """Find the atom type for a given atom name list."""
        atom_type_list = []
        for atom in atom_name_list:
            if atom in residue_definition:
                atom_type = residue_definition[atom].atom_type
            elif atom in residue_definition.removed_atom_dict:
                atom_type = residue_definition.removed_atom_dict[atom].atom_type
            elif atom == 'BLNK':
                ## TODO: use logging to log this
                return None # ic containing BLNK atom should be ignored
            else:
                raise ValueError(
                    f'Atom {atom} not found in residue definition: {residue_definition}'
                )
            atom_type_list.append(atom_type)
        return atom_type_list
    
    def res_def_fill_ic(self, residue_definition, preserve):
        """Fill in the missing parameters for the internal coordinates table
        of a residue definition."""
        for atom_key, ic_table in residue_definition.ic.items():
            atom_key = [atom.lstrip('+').lstrip('-') for atom in atom_key]
            atom_types = self._find_atom_type(atom_key, residue_definition)
            if atom_types is None:
                # ic containing BLNK atom should be ignored
                continue
            for ic_type in ic_table:
                if ic_type == 'Phi' or ic_type == 'improper':
                    continue
                if (ic_table[ic_type] is not None) and preserve:
                    continue
                ids = self.ic_position_dict[ic_type]
                cur_ic_atom_types = tuple(atom_types[i] for i in ids)
                ic_param = self._find_ic_param(cur_ic_atom_types)
                if ic_param is not None:
                    ic_table[ic_type] = ic_param

    def _find_ic_param(self, key):
        if len(key) == 2:
            bond_param = self.get_bond(key)
            if bond_param is not None:
                return bond_param.b0
            return None
        else:
            angle_param = self.get_angle(key)
            if angle_param is not None:
                return angle_param.theta0
            return None

    def fill_ic(self, topology_loader, preserve):
        """Fill in the missing parameters for the internal coordinates table
        of a topology."""
        for residue_definition in topology_loader.residues:
            self.res_def_fill_ic(residue_definition, preserve)
        # Also fill in the patched residue definitions
        for residue_definition in topology_loader.patched_defs.values():
            self.res_def_fill_ic(residue_definition, preserve)

class ResidueTopologySet:
    """Class for loading topology definition to the residue and find any missing atoms.
    Any HIS will be renamed as HSD for protein."""
    def __init__(self, entity_type=None, rtf_parser=None):
        self.rtf_version = None
        self.res_defs = {}
        self.residues = []
        self.patches = []
        self.patched_defs = {}
        self._raw_data_strings = []    
        self.entity_type = entity_type.lower()
        
        if rtf_parser is not None:
            self.load_data_dict(rtf_parser.topo_dict, rtf_parser.rtf_version)
            return
        if self.entity_type not in rtf_path_dict:
            raise ValueError(f'Unknown entity type: {entity_type}')
        self.is_hetero = self.entity_type not in ('protein', 'nucleic')

        rtf = RTFParser(file_path=rtf_path_dict[self.entity_type])
        self._raw_data_strings = rtf.lines
        self.load_data_dict(rtf.topo_dict, rtf.rtf_version)

    def load_data_dict(self, topo_data_dict: dict, rtf_version:str=None):
        """Load topology data from a dictionary. The dictionary should be parsed
        from a RTF file."""
        self.rtf_version = rtf_version
        for resname, res_topo_dict in topo_data_dict.items():
            # if resname in Entities.ResidueDefinition.na_3to1:
            #     # Map 3-letter residue name to 1-letter residue name for nucleic
            #     # acids, since biopython uses 1-letter residue name for them.
            #     resname = Entities.ResidueDefinition.na_3to1[resname]

            if res_topo_dict['is_patch']:
                res_def = Entities.PatchDefinition(
                    self.rtf_version, resname, res_topo_dict, is_hetero=self.is_hetero
                )
                self.patches.append(res_def)
            else:
                res_def = Entities.ResidueDefinition(
                    self.rtf_version, resname, res_topo_dict, is_hetero=self.is_hetero
                )
                self.residues.append(res_def)

            self.res_defs[resname] = res_def

        if 'HIS' not in self.res_defs and 'HSD' in self.res_defs:
            # Map all histidines HIS to HSD
            self.res_defs['HIS'] = self.res_defs['HSD']

    def __repr__(self):
        return (
            f'<TopologyDefinitions for {self.entity_type.upper()} Ver={self.rtf_version} '
            f'Contains {len(self.residues)} RESIDUE and '
            f'{len(self.patches)} PATCH definitions>'
        )

    def __getitem__(self, __key: 'str'):
        if __key in Entities.ResidueDefinition.na_1to3:
            __key = Entities.ResidueDefinition.na_1to3[__key]
        return self.res_defs[__key]
    
    def __contains__(self, __key: 'str'):
        if __key in Entities.ResidueDefinition.na_1to3:
            __key = Entities.ResidueDefinition.na_1to3[__key]
        return __key in self.res_defs
    
    def __iter__(self):
        return iter(self.res_defs.values())
    
class CGENFFTopologySet:
    """Class for loading topology definition to the heterogen residues where 
    the topology and parameters are generated by cgenff."""
    def __init__(self):
        self.rtf_version = '36.1'
        self.mass_dict = CGENFF_MASS_TABLE
        self.res_defs = {}
        self.residues = []
        self._raw_data_strings = [] 
        self.entity_type = 'cgenff'

    def load_rtf_block(self, rtf_block: str):
        """Load topology data from a dictionary. The dictionary should be parsed
        from a RTF file."""
        rtf = RTFParser(rtf_block=rtf_block)
        self._raw_data_strings.extend(rtf.lines)
        for resname, res_atom_dict in rtf.topo_dict.items():
            for info_dict in res_atom_dict['atoms'][0].values():
                atype = info_dict['atom_type']
                info_dict['mass'], info_dict['desc'], info_dict['element'] =\
                    self.mass_dict[atype]

            res_def = Entities.ResidueDefinition(
                rtf.rtf_version, resname, res_atom_dict, is_hetero=True
            )
            self.residues.append(res_def)
            self.res_defs[resname] = res_def

    def __repr__(self):
        return (
            f'<TopologyDefinitions for {self.entity_type.upper()} Ver={self.rtf_version} '
            f'Contains {len(self.residues)} MOLECULE definitions>'
        )

    def __getitem__(self, __key: 'str'):
        return self.res_defs[__key]
    
    def __contains__(self, __key: 'str'):
        return __key in self.res_defs
    
    def __iter__(self):
        return iter(self.residues)

class CGENFFTopologyLoader:
    """Class for loading topology definition to the heterogen residues where 
    the topology and parameters are generated by cgenff."""
    def __init__(self, cgenff_path=None, save_path=None):
        self.cgenff_path = cgenff_path
        self.rdconvert = RDKitHetConverter()
        self.cgenff_topo_set = CGENFFTopologySet()
        self.toppar_blocks = {}
        self.mol2_blocks = {}
        self.save_path = save_path
        if self.save_path is not None and not os.path.exists(self.save_path):
            warnings.warn(f'Path does not exist: {self.save_path}, creating...')
            os.makedirs(self.save_path)
        self.rdkit_mols = {}
        if self.save_path is None:
            self.save_path = os.getcwd()

    ## TODO: parse parameters from the toppar block as well
    def _get_cgenff_topology(self, input_mol2_block, ligand_toppar_file=None):
        """Load mol2 block and generate topology definition. """
        if ligand_toppar_file is not None:
            with open(ligand_toppar_file, 'r', encoding='utf-8') as f:
                toppar_block = f.read()
        else:
            toppar_block = excute_cgenff(self.cgenff_path, input_mol2_block)
        toppar_lines=toppar_block.split('\n')
        rtf_end = toppar_lines.index('END')+1

        rtf_block = '\n'.join(toppar_lines[:rtf_end])
        prm_block = '\n'.join(toppar_lines[rtf_end:])
        self.cgenff_topo_set.load_rtf_block(rtf_block)
        return toppar_block

    ## TODO: need to find topology element from rtf. Currently only add 
    ## topology definition to the atom
    def generate(self, lig_res: Entities.Heterogen, ligand_toppar_file=None):
        """Generate topology definition for the ligand residue using cgenff.
        
        Parameters
        ----------
        
        lig_res: Entities.Heterogen
            The ligand residue object.
        
        ligand_toppar_file: str, optional
            The path to the ligand topology file. If not provided, the topology
            will be generated by cgenff.
        """
        rdk_mol = lig_res.rdkit_mol
        if rdk_mol is None:
            self.rdconvert.load_heterogen(lig_res)
            rdk_mol = self.rdconvert.get_mol()
            lig_res._rdkit_mol = rdk_mol
            mol2_block = self.rdconvert.get_mol2_block()

        self.rdkit_mols[lig_res.resname] = rdk_mol
        mol2_block = MolToMol2Block(rdk_mol, ligname = lig_res.resname)
        toppar_block = self._get_cgenff_topology(mol2_block, ligand_toppar_file)
        self.mol2_blocks[lig_res.resname] = mol2_block
        self.toppar_blocks[lig_res.resname] = toppar_block
        residue_definition = self.cgenff_topo_set.res_defs[lig_res.resname]
        for atom_def in residue_definition:
            atom_def_name = atom_def.name
            if atom_def_name.startswith('LP'):
                lp = atom_def.create_new_atom()
                lig_res.lone_pair_dict[atom_def_name] = lp
                lp.parent = lig_res
            if atom_def_name not in lig_res:
                raise ValueError(
                    f'Atom {atom_def_name} not found in the residue {lig_res}!'
                    'If generating topology using a provided ligand_toppar_file, '
                    'make sure the file matches the ligand.'
                )
            atom = lig_res[atom_def.name]
            atom.topo_definition = atom_def
        lig_res.topo_definition = residue_definition

    def generate_from_rdkit(self, rdkit_mol, resname):
        """Generate topology definition for the ligand residue using cgenff. A 
        ligand Residue object will be returned"""
        self.rdconvert.load_rdkit_mol(rdkit_mol, ligand_resname=resname)
        self.rdkit_mols[resname] = rdkit_mol
        ligand = self.rdconvert.get_ligand()
        mol2_block = self.rdconvert.get_mol2_block()
        toppar_block = self._get_cgenff_topology(mol2_block)
        self.mol2_blocks[resname] = mol2_block
        self.toppar_blocks[resname] = toppar_block
        residue_definition = self.cgenff_topo_set.res_defs[resname]
        for atom_def in residue_definition:
            atom_def_name = atom_def.name
            atom = ligand[atom_def_name]
            atom.topo_definition = atom_def
        ligand.topo_definition = residue_definition
        return ligand
    
    def write_all(self, pathname=None):
        """Write all the generated files to the specified directory."""
        filenames = []
        filenames.extend(self._write_data_dict(self.mol2_blocks, 'mol2', pathname))
        filenames.extend(self._write_data_dict(self.toppar_blocks, 'str', pathname))
        return filenames
    
    def write_mol2(self, ligname, filename=None):
        """Write the ligand mol2 file. Only path to the directory is needed. If 
        path name is not provided, current working directory will be used.
        The file name will be the ligand residue name."""
        if filename is None:
            pathname = self.save_path
            filename = f'{pathname}/{ligname}.mol2'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.mol2_blocks[ligname])
        return filename

    def write_ligand_str(self, ligname, filename=None):
        """Write the content of cgenff rtf block to a file. Only path to the 
        directory is needed. If path name is not provided, current working 
        directory will be used. The file name will be the ligand residue name."""
        if filename is None:
            pathname = self.save_path
            filename = f'{pathname}/{ligname}.str'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.toppar_blocks[ligname])
        return filename

    def _write_data_dict(self, data_dict, extension, pathname):
        """(PRIVATE) Write the content of data dict to a file. Only path to the 
        directory is needed. The file name will be the ligand residue name."""
        if pathname is None:
            pathname = self.save_path
        if pathname is None:
            raise ValueError('Pathname is not provided!')
        if not os.path.exists(pathname):
            warnings.warn(f'Path does not exist: {pathname}, creating...')
            os.makedirs(pathname)
        filenames = []
        for ligname, data_block in data_dict.items():
            filename = f'{pathname}/{ligname}.{extension}'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(data_block)
                f.write('\n')
            filenames.append(filename)
        return filenames

class TopologyGenerator:
    """Class for generating topology elements from the topology definition
    and parameters.
    The topology definition the parameters are loaded from the CHARMM 36 RTF and 
    PRM files.
    If the cgenff_excutable_path is provided, the topology definition and 
    parameters for the heterogen residues are generated by cgenff, and ligand mol2
    file and cgenff rtf file will be saved if cgenff_output_path is specified."""
    def __init__(self, cgenff_excutable_path=None, cgenff_output_path=None):
        """Initialize the topology generator."""
        self.res_def_dict = {}
        self.param_dict = {}
        self.cur_defs: ResidueTopologySet = None
        self.cur_param: ParameterLoader = None
        if cgenff_excutable_path is not None:
            self.save_cgenff_output = (cgenff_output_path is not None)
            self.cgenff_loader = CGENFFTopologyLoader(
                cgenff_excutable_path, cgenff_output_path)
            self.res_def_dict['cgenff'] = self.cgenff_loader.cgenff_topo_set
        else:
            self.save_cgenff_output = False
            self.cgenff_loader = None


    def _load_residue_definitions(self, chain_type: str, preserve):
        """Load topology definition from the RTF file"""
        entity_type = chain_type_def_lookup.get(chain_type)
        if entity_type is None:
            raise NotImplementedError(
                f"Topology generation on Chain type: {chain_type} is not supported yet!"
            )
        if entity_type not in self.res_def_dict:
            self.res_def_dict[entity_type] = ResidueTopologySet(entity_type)
        if entity_type not in self.param_dict:
            self.param_dict[entity_type] = ParameterLoader(entity_type)
        
        self.cur_defs = self.res_def_dict[entity_type]
        self.cur_param = self.param_dict[entity_type]
        self.cur_param.fill_ic(self.cur_defs, preserve=preserve)
        
    def _generate_residue_topology(
            self, residue: Entities.Residue, coerce = False, QUIET = False
        ):
        """Load topology definition into the residue and find any missing atoms.
        Argument:
            residue: the Residue object whose topology is to be defined
            coerce: if True, try to coerce the modified residue name to the canonical name
                    and load the canonical residue topology definition
            QUIET: if True, suppress all warnings

        Return:
            True if the residue is defined, False otherwise"""

        if isinstance(residue, Entities.DisorderedResidue):
            return self._generate_residue_topology(
                residue.selected_child, coerce=coerce, QUIET=QUIET
            )

        if residue.resname not in self.cur_defs:
            if not QUIET:
                warnings.warn(
                    f"Residue {residue.resname} is not defined in the topology file!"
                )
            if residue.topo_definition is not None:
                return True
            if not coerce:
                return False
            success = self.coerce_resname(residue, QUIET=QUIET)
            if not success:
                return False

        if residue.topo_definition is not None and not QUIET:
            warnings.warn("Topology definition already exists! Overwriting...")

        res_definition = self.cur_defs[residue.resname]
        self.apply_topo_def_on_residue(residue, res_definition, QUIET=QUIET)
        return True
    
    @staticmethod
    def apply_topo_def_on_residue(
            residue,
            res_definition,
            QUIET = False
        ):
        """Apply the topology definition to the residue"""
        residue.topo_definition = res_definition
        residue.impropers = res_definition.impropers
        residue.cmap = res_definition.cmap
        residue.H_donors = res_definition.H_donors
        residue.H_acceptors = res_definition.H_acceptors
        residue.param_desc = res_definition.desc
        TopologyGenerator._load_atom_groups(residue)
        residue.undefined_atoms = []
        for atom in residue:
            if atom.name not in res_definition:
                residue.undefined_atoms.append(atom)
                if not QUIET:
                    parent_id = (atom.parent.id[1], atom.parent.resname)
                    warnings.warn(
                        f"Atom {atom.name} from {parent_id} is not defined in "
                        "the topology file!"
                    )

    @staticmethod
    def _create_missing_atom(residue: Entities.Residue, atom_name: str)-> list:
        """Create and separate missing heavy atoms and missing hydrogen atom by atom name"""
        missing_atom : Entities.Atom = residue.topo_definition[atom_name].create_new_atom()
        missing_atom.set_parent(residue)
        if atom_name.startswith('H'):
            residue.missing_hydrogens[atom_name] = missing_atom
        else:
            residue.missing_atoms[atom_name] = missing_atom
        return missing_atom

    @staticmethod
    def _load_group_atom_topo_definition(
            residue: Entities.Residue, atom_name_list: list
        ) -> tuple:
        """Load topology definition to each atom in the residue and find any missing 
        atoms. 
        Argument:
            atom_name_list: list of atom names that are in the same group
        Return:
            atom_group: the Atom object in the group
        """
        atom_group = []
        for atom_name in atom_name_list:
            if atom_name not in residue:
                cur_atom = TopologyGenerator._create_missing_atom(residue, atom_name)
            else:
                cur_atom = residue[atom_name]
                cur_atom.topo_definition = residue.topo_definition[atom_name]
            atom_group.append(cur_atom)
        return tuple(atom_group)

    @staticmethod
    def _load_atom_groups(residue: Entities.Residue):
        residue.atom_groups = []
        residue.missing_atoms, residue.missing_hydrogens = {},{}
        atom_group_lists = residue.topo_definition.atom_groups
        for atom_names in atom_group_lists:
            cur_group = TopologyGenerator._load_group_atom_topo_definition(
                residue, atom_names
            )
            residue.atom_groups.append(cur_group)

    def coerce_resname(self, residue: Entities.Residue, QUIET = False)->bool:
        """Coerce the name of modified residue to reconstruct it as the canonical 
        one that it is based on. 

        Argument:
            residue: the residue whose name is to be coerced
        Return:
            True if the residue is a known modified residue and is successfully coerced
            False otherwise"""
        ## TODO: add examples to the docstring
        if residue.resname not in protein_letters_3to1_extended:
            return False
        # if the residue is a known modified residue,
        # coerce the residue name to reconstruct it as the
        # canonical one
        code = protein_letters_3to1_extended[residue.resname]
        new_resname = protein_letters_1to3[code]
        if not QUIET:
            warnings.warn(
                f'Coerced Residue {residue.resname} to {new_resname}'
            )
        old_resname = residue.resname
        residue.resname = new_resname
        _, resseq, icode = residue.id
        residue.id = (" ", resseq, icode)

        if residue.parent is None:
            return True

        chain = residue.parent
        if resseq in chain.het_resseq_lookup:
            chain.het_resseq_lookup.pop(resseq)

        if hasattr(chain, "reported_res"):
            # if the chain has reported_res attribute, update it
            # to avoid generating new gaps due to mismatch resnames
            reported_res = chain.reported_res
            if (resseq, old_resname) in reported_res:
                reported_res.remove((resseq, old_resname))
                reported_res.append((resseq, new_resname))
            chain.reported_res = sorted(reported_res)

        return True

    def generate_solvent(self, solvent, solvent_model, QUIET=False):
        """Generate topology elements for solvent molecules"""
        solvent.undefined_res = []
        self._load_residue_definitions('Solvent', preserve=False)
        if solvent_model not in self.cur_defs:
            raise ValueError(f'Unknown solvent model: {solvent_model}')

        self.cur_defs.res_defs['HOH'] = self.cur_defs[solvent_model]

        for residue in solvent:
            is_defined = self._generate_residue_topology(
                residue, coerce=False, QUIET=QUIET
            )
            if not is_defined:
                solvent.undefined_res.append(residue)
        if (n_undefined:=len(solvent.undefined_res)) > 0 and not QUIET:
            warnings.warn(
                f"{n_undefined} residue(s) not defined in the chain!"
            )

        self.cur_param.fill_ic(self.cur_defs, False)
        topology = HeterogenTopology()
        solvent.topology = topology.load_chain(solvent)
        self.cur_param.apply(solvent.topology)
        return topology

    def generate(
            self, chain: Entities.Chain, coerce: bool = False,
            first_patch: str = None, last_patch: str = None,
            auto_correct_first_patch=True,
            preserve_ic = True, QUIET = False
        ):
        """Load topology definition into the chain and find any missing atoms.
        Argument:
            chain: the Chain object whose topology is to be defined
            auto_correct_first_patch: if True, automatically correct the first patch
                of polypeptide chains. Since PRO or GLY needs special treatment when
                patched at the N-terminus, the first patch will be corrected to PROP
                or GLYP respectively.
            coerce: if True, try to coerce the modified residue name to the canonical name
                    and load the canonical residue topology definition
            QUIET: if True, suppress all warnings
        """
        chain.sort_residues()
        if chain.residues[0].resname == 'ACE':
            # if the first residue is ACE, remove it
            chain.truncate(start = 1)
        chain.undefined_res = []
        self._load_residue_definitions(chain.chain_type, preserve_ic)
        for residue in chain:
            is_defined = self._generate_residue_topology(
                residue, coerce=coerce, QUIET=QUIET
            )
            if not is_defined:
                chain.undefined_res.append(residue)
        if (n_undefined:=len(chain.undefined_res)) > 0 and not QUIET:
            warnings.warn(
                f"{n_undefined} residue(s) not defined in the chain!"
            )
        if first_patch is not None or last_patch is not None:
            self.patch_termini(
                chain, first_patch, last_patch, 
                auto_correct_first_patch, 
                QUIET=QUIET
            )

        self.cur_param.fill_ic(self.cur_defs, preserve_ic)
        if chain.chain_type in ('Polypeptide(L)', 'Polyribonucleotide', 'Polydeoxyribonucleotide'):
            topology = ChainTopology()
        else:
            topology = HeterogenTopology()
        chain.topology = topology.load_chain(chain)
        self.cur_param.apply(chain.topology)
        return topology

    def patch_termini(
            self, chain: Entities.PolymerChain,
            first: str, last: str, auto_correct_first_patch=True, QUIET=False
        ):
        """Patch the terminal residues of the chain"""
        if chain.chain_type not in ('Polypeptide(L)', 'Polyribonucleotide'):
            raise NotImplementedError(
                "Only polypeptide and polynucleotide chains are supported "
                f"for patching! Got {chain.chain_type}"
            )
        chain.sort_residues()
        if auto_correct_first_patch:
            corrected_first = protein_n_term_patch_correction.get(
                chain.residues[0].resname, first
            )
            if corrected_first != first and not QUIET:
                warnings.warn(
                    f"Corrected first patch from {first} to {corrected_first}"
                    f" for the N-terminal residue {chain.residues[0].resname}"
                    f" in chain {chain.id}"
                )
            first = corrected_first

        if first is not None:
            self.patch_residue(
                chain.child_list[0], first, patch_loc="NTER", QUIET=QUIET
            )
        if last is not None:
            self.patch_residue(
                chain.child_list[-1], last, patch_loc="CTER", QUIET=QUIET
            )
        if chain.topology is not None:
            # Update topology elements if they are already defined
            chain.topology.update()

    def patch_residue(
            self, residue: Entities.Residue, patch: str,
            patch_loc = 'MIDCHAIN', QUIET = False
        ):
        """Patch the residue with the patch definition"""
        if residue.topo_definition is None:
            raise ValueError(
                f"Cannot patch the first residue {residue.resname} "
                "because it is undefined (no topology definition exists)!"
            )
        res_def = residue.topo_definition
        patched_def_name = res_def.resname + "_" + patch
        # if patched_def_name in self.cur_defs.patched_defs:
        #     patched_res_def = self.cur_defs.patched_defs[patched_def_name]
        # else:
        #     patcher = ResiduePatcher()
        #     patch_def = self.cur_defs[patch]
        #     patched_res_def = patcher.patch_residue_definition(
        #         res_def, patch_def, patch_loc=patch_loc
        #     )
        #     self.cur_defs.patched_defs[patched_def_name] = patched_res_def

        # find the right residue definition set ('protein', 'nucleic', 'water_ions', etc)
        for res_def_container in self.res_def_dict.values():
            if residue.resname in res_def_container:
                break
        if patched_def_name in res_def_container.patched_defs:
            # if the patched residue definition already exists, use it
            patched_res_def = res_def_container.patched_defs[patched_def_name]
        else:
            # otherwise, create a new patched residue definition and store it
            patcher = ResiduePatcher()
            patch_def = res_def_container[patch]
            patched_res_def = patcher.patch_residue_definition(
                res_def, patch_def, patch_loc=patch_loc
            )
            res_def_container.patched_defs[patched_def_name] = patched_res_def
        
        self.apply_topo_def_on_residue(residue, patched_res_def, QUIET=QUIET)
        fixer = ResidueFixer()
        fixer.load_residue(residue)
        fixer.remove_undefined_atoms()
    
    def generate_model(
            self, model: Entities.OrganizedModel.OrganizedModel, coerce: bool = False,
            prot_first_patch: str = 'ACE', prot_last_patch: str = 'CT3',
            na_first_patch: str = '5TER', na_last_patch: str = '3PHO',
            auto_correct_first_patch=True, build_coords = True,
            preserve_ic = True, solvent_model = 'TIP3', 
            QUIET = False
        ):
        """Generate topology for a organized model. Protein, nucleic acid, ion and solvent will
        be generated separately. If cgenff executable path is provided, cgenff will be used to
        generate topology for the ligand, nucleoside phosphate, and co-solvent residues.
        Argument:
            model: the OrganizedModel object whose topology is to be defined
            coerce: if True, try to coerce the modified residue name to the canonical name
                    and load the canonical residue topology definition
            prot_first_patch: the patch to be applied to the N-terminal residue of the protein.
                Default is ACE (acetylated N-terminus)
            prot_last_patch: the patch to be applied to the C-terminal residue of the protein.
                Default is CT3 (N-Methylamide C-terminus)
            na_first_patch: the patch to be applied to the 5' terminal residue of the nucleic acid
                Default is 5TER (5'-terminal HYDROXYL)
            na_last_patch: the patch to be applied to the 3' terminal residue of the nucleic acid
                Default is 3PHO (3'-terminal PHOSPHATE)
            auto_correct_first_patch: if True, automatically correct the first patch
                of polypeptide chains. Since PRO or GLY needs special treatment when
                patched at the N-terminus, the first patch will be corrected to PROP
                or GLYP respectively.
            build_coords: if True, build the coordinates of the missing atoms
            preserve_ic: if True, preserve the internal coordinates of the residue
            solvent_model: the solvent model to be used for the solvent molecules.
                default is TIP3
            QUIET: if True, suppress all warnings
        """
        if not isinstance(model, Entities.OrganizedModel.OrganizedModel):
            raise TypeError(
                "Invalid argument type provided! OrganizedModel"
                f' is expected. {type(model)} is provided.'
            )
        for chain in model.protein:
            self.generate(
                chain, coerce=coerce, first_patch=prot_first_patch,
                last_patch=prot_last_patch,
                auto_correct_first_patch=auto_correct_first_patch,
                preserve_ic=preserve_ic, QUIET=QUIET
            )
            if build_coords:
                fix_chain(chain)
        for chain in (model.RNA + model.DNA):
            self.generate(
                chain, coerce=coerce, first_patch=na_first_patch,
                last_patch=na_last_patch, auto_correct_first_patch=False,
                preserve_ic=preserve_ic, QUIET=QUIET
            )
            if build_coords:
                fix_chain(chain)
        for chain in model.ion:
            self.generate(
                chain, coerce=coerce, first_patch=None,
                last_patch=None, auto_correct_first_patch=False,
                preserve_ic=False, QUIET=QUIET
            )
        for chain in model.solvent:
            self.generate_solvent(chain, solvent_model, QUIET=QUIET)

        if self.cgenff_loader is not None:
            for chain in model.ligand+model.co_solvent+model.phos_ligand:
                for lig_residue in chain:
                    self.cgenff_loader.generate(lig_residue)
            if self.save_cgenff_output:
                self.cgenff_loader.write_all()
        model.topology_loader = self
        model.topology = ModelTopology(model)

##TODO: Rewrite TopoDef and this class to use the Topology instead
## of removing entries one by one here
class ResiduePatcher:
    """Class Object for patching a residue with a patch definition"""
    def __init__(self):
        self.res: Entities.ResidueDefinition = None
        self.patch: Entities.PatchDefinition = None
        # for marking the ic entry to be removed from terminal residue during 
        # patching, either '+' or '-' depending on the patch location
        self.remove_nei_ic_prefix: str = None

    def _remove_atom_from_bonds(self, atom_name:str):
        """Remove the atom from the bonds"""
        for bonds in self.res.bonds.values():
            remove_list = []
            for bond in bonds:
                if atom_name in bond:
                    remove_list.append(bond)
            for bond in remove_list:
                bonds.remove(bond)
        
    def _remove_neighbor_atom_from_ic(self):
        remove_keys = set()
        for ic_key in self.res.ic:
            for atom in ic_key:
                if atom.startswith(self.remove_nei_ic_prefix):
                    remove_keys.add(ic_key)
        for ic_key in remove_keys:
            self.res.ic.pop(ic_key)

    def _remove_neighbor_atom_from_cmap(self):
        remove_keys = set()
        for cmap in self.res.cmap:
            for dihe in cmap:
                for atom in dihe:
                    if atom.startswith(self.remove_nei_ic_prefix):
                        remove_keys.add(cmap)

        for cmap in remove_keys:
            self.res.cmap.remove(cmap)

    def _remove_neighbor_atom_from_improper(self):
        remove_keys = set()
        for improper in self.res.impropers:
            for atom in improper:
                if atom.startswith(self.remove_nei_ic_prefix):
                    remove_keys.add(improper)

        for improper in remove_keys:
            self.res.impropers.remove(improper)

    def _remove_atom_from_ic(self, atom_name:str):
        """Remove the atom from the ic table dictionary"""
        remove_keys = []
        for ic_atom_names in self.res.ic:
            if atom_name in ic_atom_names:
                remove_keys.append(ic_atom_names)
        for key in remove_keys:
            self.res.ic.pop(key)

    def _remove_atom_from_param(self, param_attr, atom_name:str):
        """Remove the residue from the parameter attribute of the residue"""
        remove_keys = set()
        for iterable in param_attr:
            if atom_name in iterable:
                remove_keys.add(iterable)
        
        for iterable in remove_keys:
            param_attr.remove(iterable)

    def _remove_atom_from_cmap(self, cmaps, atom_name:str):
        """Remove the residue from the parameter attribute of the residue"""
        remove_keys = set()
        for cmap in cmaps:
            for dihe in cmap:
                if atom_name in dihe:
                    remove_keys.add(cmap)

        for cmap in remove_keys:
            cmaps.remove(cmap)

    def _delete_atom_params(self):
        """Delete the parameters of the atoms that are deleted in the patch"""
        if self.patch.delete is None:
            return
        delete_atom_names = []
        for entity_type, entity_name in self.patch.delete:
            if entity_type == 'ATOM' and entity_name in self.res:
                delete_atom_names.append(entity_name)
        for atom_name in delete_atom_names:
            self.res.removed_atom_dict[atom_name] = self.res.atom_dict.pop(atom_name)
            self._remove_atom_from_bonds(atom_name)
            self._remove_atom_from_ic(atom_name)
            for param_attr in (
                self.res.impropers, self.res.H_donors,
                self.res.H_acceptors, self.res.atom_groups
            ):
                self._remove_atom_from_param(param_attr, atom_name)
            self._remove_atom_from_cmap(self.res.cmap, atom_name)

    def _apply_patch(self):
        """Apply the patch on the residue definition"""
        self._delete_atom_params()
        self.res.atom_dict.update(self.patch.atom_dict)
        for bond_type in self.res.bonds:
            self.res.bonds[bond_type].extend(self.patch.bonds[bond_type])

        self.res.atom_groups.extend(self.patch.atom_groups)
        self.res.H_donors.extend(self.patch.H_donors)
        self.res.H_acceptors.extend(self.patch.H_acceptors)
        self.res.impropers.extend(self.patch.impropers)
        self.res.cmap.extend(self.patch.cmap)

        self.res.ic = {**self.res.ic, **self.patch.ic}
        if self.remove_nei_ic_prefix is not None:
            self._remove_neighbor_atom_from_ic()
            self._remove_neighbor_atom_from_improper()
            self._remove_neighbor_atom_from_cmap()

        self.res.is_modified = True

    def patch_residue_definition(
            self,
            residue_definition,
            patch_definition,
            patch_loc: str = "MIDCHAIN",
        ):
        """Patch a residue definition with a patch definition. Return the patched residue definition
        """
        patch_loc = patch_loc.upper()
        if patch_loc not in ("MIDCHAIN", "NTER", "CTER"):
            raise ValueError("Patch type must be 'MIDCHAIN', 'NTER' or 'CTER'")
        if patch_loc == "NTER":
            self.remove_nei_ic_prefix = '-'
        elif patch_loc == "CTER":
            self.remove_nei_ic_prefix = '+'

        # need to make a copy of the patch definition
        # to avoid modifying the original patch definition
        self.res = deepcopy(residue_definition)
        # remove the standard residue coordinates
        self.res.standard_coord_dict = None
        self.res._standard_res = None
        self.patch = deepcopy(patch_definition)
        self._apply_patch()
        self.res.assign_donor_acceptor()
        self.res.create_atom_lookup_dict()
        self.res.patch_with = self.patch.resname
        return self.res
    
    def patch_disulfide(
            self, res1: Entities.ResidueDefinition, res2: Entities.ResidueDefinition
        ):
        """Patch the disulfide bond between two cysteine residues"""
        ## Disulfide bond patching is hard coded. The residue and atom definitions
        ## are modified here directly without using the DISU patch definition.
        ## Dihedrals and impropers are not generated.
        ## TODO: fully implement the DISU patch definition to the topology definition
        ## and parameters
        if not isinstance(res1, Entities.ResidueDefinition) or not isinstance(res2, Entities.ResidueDefinition):
            raise TypeError("res1 and res2 must be ResidueDefinition objects")
        if res1.resname != 'CYS' or res2.resname != 'CYS':
            raise ValueError("res1 and res2 must be CYS residues")
        if res1.is_modified or res2.is_modified:
            raise ValueError("res1 and res2 must not be modified")
        
        self.res = deepcopy(res1)
        remove_name = 'HG1'
        self.res.removed_atom_dict[remove_name] = self.res.atom_dict.pop(remove_name)
        self._remove_atom_from_bonds(remove_name)
        self._remove_atom_from_ic(remove_name)
        for param_attr in (
            self.res.impropers, self.res.H_donors,
            self.res.H_acceptors, self.res.atom_groups
        ):
            self._remove_atom_from_param(param_attr, remove_name)
        self._remove_atom_from_cmap(self.res.cmap, remove_name)
        self.res.assign_donor_acceptor()
        self.res.create_atom_lookup_dict()
        self.res.patch_with = 'DISU'
        sulfur_def = self.res['SG']
        sulfur_def.atom_type = 'SM'
        sulfur_def.charge = -0.08
        CB_def = self.res['CB']
        CB_def.charge = -0.10

        # for bond_type in self.res.bonds:
        #     self.res.bonds[bond_type].extend(('SG1','SG2'))

        # self.res.atom_groups.extend(self.patch.atom_groups)

        # self.res.ic = {**self.res.ic, **self.patch.ic}
        # if self.remove_nei_ic_prefix is not None:
        #     self._remove_neighbor_atom_from_ic()
        #     self._remove_neighbor_atom_from_improper()
        #     self._remove_neighbor_atom_from_cmap()

        self.res.is_modified = True

        return self.res