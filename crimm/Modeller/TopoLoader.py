import warnings
import pickle
from copy import deepcopy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_1to3
from crimm import StructEntities as Entities
from crimm.IO.RTFParser import RTFParser

def _find_atom_in_residue(residue, atom_name):
    if atom_name.startswith('+') or atom_name.startswith('-'):
        return
    if atom_name in residue:
        return residue[atom_name]
    if atom_name.startswith('H'):
        return residue.missing_hydrogens[atom_name]
    return residue.missing_atoms[atom_name]

def get_bonds_within_residue(residue):
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
                continue
            bonds.append(
                Entities.Bond(atom1, atom2, bond_type)
            )
    return bonds

def atom_add_neighbors(atom1, atom2):
    """Add atom2 to atom1's neighbors and vice versa"""
    atom1.neighbors.add(atom2)
    atom2.neighbors.add(atom1)

def residue_trace_atom_neigbors(residue):
    """Trace all bonds within the residue and add the atoms to each other's
    neighbors list. Return a list of bonds within the residue."""
    bonds = get_bonds_within_residue(residue)
    for bond in bonds:
        a1, a2 = bond
        atom_add_neighbors(a1, a2)
    return bonds

def chain_trace_atom_neighbors(chain, inter_res_bonding_atoms, bond_type='single'):
    """Trace all bonds within the chain and add the atoms to each other's
    neighbors list. Return a list of bonds within the chain."""
    end_atom, start_atom = inter_res_bonding_atoms
    # e.g. this would be C-N for peptide, 
    # thus end_atom is C and start_atom is N
    chain.sort_residues()
    all_bonds = []
    for i, cur_res in enumerate(chain.residues[:-1]):
        next_res = chain.residues[i+1]
        bonds = residue_trace_atom_neigbors(cur_res)
        all_bonds.extend(bonds)
        # add inter-residue neigbors from peptide/nucleotide bond
        if (end_atom in cur_res) and (start_atom in next_res):
            a1, a2 = cur_res[end_atom], next_res[start_atom]
            atom_add_neighbors(a1, a2)
            inter_res_bond = Entities.Bond(a1, a2, bond_type)
            all_bonds.append(inter_res_bond)
        else:
            print(start_atom, end_atom, 'not exists')
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

class TopologyLoader:
    """Class for loading topology definition to the residue and find any missing atoms."""
    def __init__(self, file_path=None, data_dict_path=None):
        self.rtf_version = None
        self.res_defs = {}
        self.residues = []
        self.patches = []

        if data_dict_path is not None:
            with open(data_dict_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.load_data_dict(data_dict)
        elif file_path is not None:
            rtf = RTFParser(file_path=file_path)
            self.load_data_dict(rtf.topo_dict, rtf.rtf_version)

    def load_data_dict(self, topo_data_dict, rtf_version=None):
        """Load topology data from a dictionary. The dictionary should be parsed
        from a RTF file."""
        self.rtf_version = rtf_version
        for resname, res_topo_dict in topo_data_dict.items():
            if res_topo_dict['is_patch']:
                res_def = Entities.PatchDefinition(
                    self.rtf_version, resname, res_topo_dict
                )
                self.patches.append(res_def)
            else:
                res_def = Entities.ResidueDefinition(
                    self.rtf_version, resname, res_topo_dict
                )
                self.residues.append(res_def)
            self.res_defs[resname] = res_def

        if 'HIS' not in self.res_defs and 'HSD' in self.res_defs:
            # Map all histidines HIS to HSD
            self.res_defs['HIS'] = self.res_defs['HSD']

         

    def __repr__(self):
        return (
            f'<TopologyLoader Ver={self.rtf_version} '
            f'Contains {len(self.residues)} RESIDUE and '
            f'{len(self.patches)} PATCH definitions>'
        )

    def __getitem__(self, __key: 'str'):
        return self.res_defs[__key]
    
    def __iter__(self):
        return iter(self.res_defs.values())
    
    def generate_residue_topology(
            self, residue: Entities.Residue, coerce: bool = False, QUIET=False
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
            return self.generate_residue_topology(
                residue.selected_child, coerce=coerce, QUIET=QUIET
            )

        if residue.resname not in self.res_defs:
            if not QUIET:
                warnings.warn(
                    f"Residue {residue.resname} is not defined in the topology file!"
                )
            if residue.topo_definition is not None:
                return True
            if not coerce:
                return False
            return self.coerce_resname(residue, QUIET=QUIET)

        if residue.topo_definition is not None and not QUIET:
            warnings.warn("Topology definition already exists! Overwriting...")
        
        res_definition = self.res_defs[residue.resname]
        residue.topo_definition = res_definition
        residue.total_charge = res_definition.total_charge
        residue.impropers = res_definition.impropers
        residue.cmap = res_definition.cmap
        residue.H_donors = res_definition.H_donors
        residue.H_acceptors = res_definition.H_acceptors
        residue.param_desc = res_definition.desc
        self._load_atom_groups(residue)
        residue.undefined_atoms = []
        for atom in residue:
            if atom.name not in res_definition:
                residue.undefined_atoms.append(atom)
                if not QUIET:
                    warnings.warn(
                        f"Atom {atom.name} is not defined in the topology file!"
                    )
        return True
    
    @staticmethod
    def _create_missing_atom(residue: Entities.Residue, atom_name: str):
        """Create and separate missing heavy atoms and missing hydrogen atom by atom name"""
        missing_atom = residue.topo_definition[atom_name].create_new_atom()
        missing_atom.set_parent(residue)
        if atom_name.startswith('H'):
            residue.missing_hydrogens[atom_name] = missing_atom
        else:
            residue.missing_atoms[atom_name] = missing_atom

    @staticmethod
    def _load_group_atom_topo_definition(
            residue: Entities.Residue, atom_name_list
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
                TopologyLoader._create_missing_atom(residue, atom_name)
                continue
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
            cur_group = TopologyLoader._load_group_atom_topo_definition(
                residue, atom_names
            )
            residue.atom_groups.append(cur_group)

    def coerce_resname(self, residue: Entities.Residue, QUIET=False):
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
        residue.resname = new_resname
        _, resseq, icode = residue.id
        residue.id = (" ", resseq, icode)
        if hasattr(residue.parent, "reported_res"):
            # if the chain has reported_res attribute, update it
            # to avoid generating new gaps due to mismatch resnames
            residue.parent.reported_res[resseq-1] = (resseq, new_resname)
        return True

    def generate_chain_topology(
            self, chain: Entities.Chain, coerce: bool = False, QUIET=False
        ):
        """Load topology definition into the chain and find any missing atoms.
        Argument:
            chain: the Chain object whose topology is to be defined
            coerce: if True, try to coerce the modified residue name to the canonical name
                    and load the canonical residue topology definition
            QUIET: if True, suppress all warnings
        """
        chain.undefined_res = []
        for residue in chain:
            is_defined = self.generate_residue_topology(residue, coerce=coerce, QUIET=QUIET)
            if not is_defined:
                chain.undefined_res.append(residue)
        if (n_undefined:=len(chain.undefined_res)) >0 and not QUIET:
            warnings.warn(
                f"{n_undefined} residues are not defined in the chain!"
            )
        topo_elements = TopologyElementContainer()
        chain.topo_elements = topo_elements.load_chain(chain)


class TopologyElementContainer:
    """A class object that stores topology elements"""
    def __init__(self):
        self._visited_atoms = None
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None
        self.cmap = None
        self.nonbonded = None
        self.atom_lookup = None
        self.missing_param_dict = None
        self.containing_entity = None

    def __iter__(self):
        topo_attrs = [
            'bonds', 'angles', 'dihedrals', 'impropers', 'cmap', 'nonbonded'
        ]
        for attr in topo_attrs:
            yield attr, getattr(self, attr)

    def __repr__(self) -> str:
        if self.containing_entity is None:
            return "<EmptyTopologyElementContainer>"
        s = f"<TopologyElementContainer for {self.containing_entity} with "
        for attr, value in self:
            if value is None:
                n = 0
            else:
                n = len(value)
            s += f"{attr}={n}, "
        s = s[:-2] + ">"
        return s
        
    def load_chain(self, chain: Entities.Chain):
        """Find all topology elements from the chain"""
        self.containing_entity = chain
        self.find_topo_elements(chain)
        self.create_atom_lookup_table()
        return self
    
    ## TODO: get Improper, Cmap, and Nonbonded from the topology
    def find_topo_elements(self, chain: Entities.Chain):
        """Find all topology elements in the chain"""
        if chain.chain_type == 'Polypeptide(L)':
            inter_res_bond = ('C','N') # peptide bond
        elif chain.chain_type == 'Polyribonuleotide':
            inter_res_bond = ('C3\'','P')
        else:
            raise NotImplementedError("Chain type not supported yet!")
        
        self.bonds = chain_trace_atom_neighbors(chain, inter_res_bond)
        visited_atoms = set()
        angles = set()
        dihedrals = set()
        first_atom = chain.residues[0].atoms[0]
        traverse_graph(first_atom, angles, dihedrals, visited_atoms)
        self._visited_atoms = list(visited_atoms)
        self.angles = list(angles)
        self.dihedrals = list(dihedrals)
    
    def create_atom_lookup_table(self) -> dict:
        """Create a lookup table for all topology elements for a given atom in the chain"""
        atom_lookup = {
            atom:{
                'bonds': [],
                'angles': [],
                'dihedrals': []
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

        self.atom_lookup = atom_lookup

class ResiduePatcher:
    """Class Object for patching a residue with a patch definition"""
    def __init__(self):
        self.res = None
        self.patch = None

    def _remove_atom_from_param(self, param_attr, atom_name:str):
        """Remove the residue from the parameter attribute of the residue"""
        for iterable in param_attr:
            if atom_name in iterable:
                param_attr.remove(iterable)

    def _delete_atom_params(self):
        """Delete the parameters of the atoms that are deleted in the patch"""
        if self.patch.delete is None:
            return
        for entity_type, entity_name in self.patch.delete:
            if entity_type == 'ATOM' and entity_name in self.res:
                for param_attr in (
                    self.res.impropers, self.res.cmap, self.res.H_donors, 
                    self.res.H_acceptors, self.res.atom_groups
                ):
                    self._remove_atom_from_param(param_attr, entity_name)
                remove_keys = []
                for atom_names in self.res.ic:
                    if entity_name in atom_names:
                        remove_keys.append(atom_names)
        for key in remove_keys:
            self.res.ic.pop(key)
                
    def _apply_patch(self):
        """Apply the patch on the residue definition"""
        self.res.atom_dict.update(self.patch.atom_dict)
        for bond_type in self.res.bonds:
            self.res.bonds[bond_type].extend(self.patch.bonds[bond_type])

        self.res.total_charge = 0
        for atom_def in self.res:
            self.res.total_charge += atom_def.charge

        self._delete_atom_params()
        new_ic = {**self.res.ic, **self.patch.ic}
        if ('CY', 'CA', 'N', 'HN') in self.patch.ic:
            new_ic.pop(('-C', 'CA', 'N', 'HN'))
        self.res.ic = new_ic.copy()

    def patch_residue_definition(
            self,
            residue_definition,
            patch_definition
        ):
        """Patch a residue definition with a patch definition. Return the patched 
        residue definition
        """
        self.res = deepcopy(residue_definition)
        # remove the standard residue coordinates
        self.res.standard_coord_dict = None
        self.res._standard_res = None
        self.patch = deepcopy(patch_definition)
        self._apply_patch()
        self.res.assign_donor_acceptor()
        self.res.create_atom_lookup_dict()
        return self.res