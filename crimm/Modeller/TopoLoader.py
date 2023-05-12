import os
import warnings
import pickle
from typing import List, Tuple
from copy import deepcopy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_1to3
from crimm import StructEntities as Entities
from crimm.IO.RTFParser import RTFParser
from crimm.Modeller import ResidueFixer

toppar_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../Data/toppar')
)
rtf_path_dict = {
    "protein": os.path.join(toppar_dir, 'prot.rtf'),
    "nucleic": os.path.join(toppar_dir, 'na.rtf'),
    "lipid": os.path.join(toppar_dir, 'lipid.rtf'),
    "carb": os.path.join(toppar_dir, 'carb.rtf'),
    "ethers": os.path.join(toppar_dir, 'ethers.rtf'),
    "cgenff": os.path.join(toppar_dir, 'cgenff.rtf')
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

def chain_clear_atom_neighbors(chain: Entities.PolymerChain):
    """Clear all neighbors of atoms in the chain"""
    for atom in chain.get_atoms():
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
    chain_clear_atom_neighbors(chain)
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

##TODO: separate this class into two classes: TopologyLoader and TopologyGenerator
class TopologyLoader:
    """Class for loading topology definition to the residue and find any missing atoms."""
    def __init__(self, entity_type=None, data_dict_path=None):
        self.rtf_version = None
        self.res_defs = {}
        self.residues = []
        self.patches = []
        self.patched_defs = {}
        self._raw_data_strings = []

        if data_dict_path is not None:
            with open(data_dict_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.load_data_dict(data_dict)

        elif entity_type is not None:
            entity_type = entity_type.lower()
            if entity_type not in rtf_path_dict:
                raise ValueError(f'Unknown entity type: {entity_type}')
            rtf = RTFParser(file_path=rtf_path_dict[entity_type])
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
        if __key in Entities.ResidueDefinition.na_1to3:
            __key = Entities.ResidueDefinition.na_1to3[__key]
        return self.res_defs[__key]
    
    def __contains__(self, __key: 'str'):
        if __key in Entities.ResidueDefinition.na_1to3:
            __key = Entities.ResidueDefinition.na_1to3[__key]
        return __key in self.res_defs
    
    def __iter__(self):
        return iter(self.res_defs.values())
    
    def generate_residue_topology(
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
            return self.generate_residue_topology(
                residue.selected_child, coerce=coerce, QUIET=QUIET
            )

        if residue.resname not in self:
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

        res_definition = self[residue.resname]
        self.apply_topo_def_on_residue(residue, res_definition, QUIET=QUIET)
        return True

    @staticmethod
    def apply_topo_def_on_residue(
            residue: Entities.Residue,
            res_definition: Entities.ResidueDefinition,
            QUIET = False
        ):
        """Apply the topology definition to the residue"""
        residue.topo_definition = res_definition
        residue.total_charge = res_definition.total_charge
        residue.impropers = res_definition.impropers
        residue.cmap = res_definition.cmap
        residue.H_donors = res_definition.H_donors
        residue.H_acceptors = res_definition.H_acceptors
        residue.param_desc = res_definition.desc
        TopologyLoader._load_atom_groups(residue)
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
                cur_atom = TopologyLoader._create_missing_atom(residue, atom_name)
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
            cur_group = TopologyLoader._load_group_atom_topo_definition(
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
        residue.resname = new_resname
        _, resseq, icode = residue.id
        residue.id = (" ", resseq, icode)

        if residue.parent is not None:
            residue.parent.het_res.remove(residue)
            residue.parent.het_resseq_lookup.pop(resseq)

        if hasattr(residue.parent, "reported_res"):
            # if the chain has reported_res attribute, update it
            # to avoid generating new gaps due to mismatch resnames
            residue.parent.reported_res[resseq-1] = (resseq, new_resname)
        return True

    def generate_chain_topology(
            self, chain: Entities.Chain, coerce: bool = False,
            first_patch: str = None, last_patch: str = None,
            QUIET = False
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
            is_defined = self.generate_residue_topology(
                residue, coerce=coerce, QUIET=QUIET
            )
            if not is_defined:
                chain.undefined_res.append(residue)
        if (n_undefined:=len(chain.undefined_res)) > 0 and not QUIET:
            warnings.warn(
                f"{n_undefined} residue(s) not defined in the chain!"
            )
        if first_patch is not None or last_patch is not None:
            self.patch_termini(chain, first_patch, last_patch, QUIET=True)

        topo_elements = TopologyElementContainer()
        chain.topo_elements = topo_elements.load_chain(chain)

    def patch_termini(
            self, chain: Entities.PolymerChain,
            first: str, last: str, QUIET=False
        ):
        """Patch the terminal residues of the chain"""
        if chain.chain_type not in ('Polypeptide(L)', 'Polyribonucleotide'):
            raise NotImplementedError(
                "Only polypeptide and polynucleotide chains are supported "
                f"for patching! Got {chain.chain_type}"
            )
        chain.sort_residues()
        if first is not None:
            self.patch_residue(chain.child_list[0], first, "NTER", QUIET=QUIET)
        if last is not None:
            self.patch_residue(chain.child_list[-1], last, "CTER", QUIET=QUIET)
        if chain.topo_elements is not None:
            # Update topology elements if they are already definedß
            topo_elements = TopologyElementContainer()
            chain.topo_elements = topo_elements.load_chain(chain)

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
        if patched_def_name in self.patched_defs:
            patched_res_def = self.patched_defs[patched_def_name]
        else:
            patcher = ResiduePatcher()
            patch_def = self[patch]
            patched_res_def = patcher.patch_residue_definition(
                res_def, patch_def, patch_loc=patch_loc
            )
            self.patched_defs[patched_def_name] = patched_res_def

        self.apply_topo_def_on_residue(residue, patched_res_def, QUIET=QUIET)
        fixer = ResidueFixer()
        fixer.load_residue(residue)
        fixer.remove_undefined_atoms()


class TopologyElementContainer:
    """A class object that stores topology elements"""
    def __init__(self):
        self._visited_atoms = None
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None
        self.cmap = None
        self.atom_lookup = None
        self.missing_param_dict = None
        self.containing_entity = None

    def __iter__(self):
        topo_attrs = [
            'bonds', 'angles', 'dihedrals', 'impropers', 'cmap'
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

    def load_chain(self, chain: Entities.PolymerChain):
        """Find all topology elements from the chain"""
        self.containing_entity = chain
        self.find_topo_elements(chain)
        self.create_atom_lookup_table()
        return self
    
    @staticmethod
    def _find_seeding_atom(chain):
        """Find the first atom to start the search"""
        for atom in chain.child_list[0]:
            if len(atom.neighbors) > 0:
                return atom

    ## TODO: get Cmap from the topology rtf file
    def find_topo_elements(self, chain: Entities.Chain):
        """Find all topology elements in the chain"""
        if chain.chain_type == 'Polypeptide(L)':
            inter_res_bond = ('C','N') # peptide bond
        elif chain.chain_type == 'Polyribonucleotide':
            inter_res_bond = ("O3'",'P') # phosphodiester bond
        else:
            raise NotImplementedError("Chain type not supported yet!")
        
        if chain.undefined_res:
            raise ValueError(
                "Cannot find topology elements for a chain with undefined residues! "
                f"Undefined residues: {chain.undefined_res}. This is possibly due "
                "to heterogen residues. Use Chain.generate_chain_topology() "
                "with coerce=True to coerce the heterogen into canoncal residue "
                "if necessary."
            )

        if not chain.is_continuous():
            raise ValueError(
                "Cannot generate topology elements for a chain with "
                f"discontinuous backbone! Missing residues: {chain.gaps}. Use "
                "loop modeling tool to construct the missing residues first."
            )

        self.bonds = chain_trace_atom_neighbors(chain, inter_res_bond)
        visited_atoms = set()
        angles = set()
        dihedrals = set()
        seed_atom = self._find_seeding_atom(chain)
        traverse_graph(seed_atom, angles, dihedrals, visited_atoms)
        self._visited_atoms = list(visited_atoms)
        self.angles = list(angles)
        self.dihedrals = list(dihedrals)
        self.impropers = get_impropers(chain)

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

##TODO: Rewrite TopoDef and this class to use the TopologyElementContainer instead
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

        self.res.total_charge = 0
        for atom_def in self.res:
            self.res.total_charge += atom_def.charge
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
            residue_definition: Entities.ResidueDefinition,
            patch_definition: Entities.PatchDefinition,
            patch_loc: str = "MIDCHAIN",
        ) -> Entities.ResidueDefinition:
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