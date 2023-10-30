import warnings
from collections import OrderedDict
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation as R

def sep_by_priorities(atom_list):
    # Create a 2-tier priotity list to separate
    # heavy atoms and hydrogen atom
    missing_atoms, missing_hydrogens = [],[]
    for atom in atom_list:
        if atom.startswith('H'):
            missing_hydrogens.append(atom)
        else:
            missing_atoms.append(atom)
    return missing_atoms, missing_hydrogens
    
def recur_find_build_seq(
    running_dict: OrderedDict, missing_atoms, build_seq, exclude_list
):

    if len(missing_atoms) == 0:
        return build_seq
 
    atom_name, ic_list = running_dict.popitem(last=False)
    for ic_keys in ic_list:
        cur_ic_set = set(ic_keys)
        if (len(cur_ic_set.intersection(missing_atoms)) > 1) or (
            cur_ic_set.intersection(exclude_list)
        ):
            continue

        missing_atoms.remove(atom_name)
        build_seq.append((atom_name, ic_keys))
        return recur_find_build_seq(
            running_dict, missing_atoms, build_seq, exclude_list
        )

    running_dict[atom_name] = ic_list
    return recur_find_build_seq(
        running_dict, missing_atoms, build_seq, exclude_list
    )

def get_coord_from_dihedral_ic(i_coord, j_coord, k_coord, phi, t_jkl, r_kl):
    origin = j_coord
    a1 = i_coord - origin
    a2 = k_coord - origin
    r1 = R.from_rotvec(phi * a2/LA.norm(a2), degrees=True)
    n1 = np.cross(a1, a2)
    n2 = r1.apply(n1)
    m = np.cross(a2, n2)
    r2 = R.from_rotvec((t_jkl-90) * n2/LA.norm(n2), degrees=True)
    a3 = r2.apply(m)
    a3 = a3/LA.norm(a3)*r_kl

    l_coord = origin+a2+a3
    return l_coord
    
def get_coord_from_improper_ic(i_coord, j_coord, k_coord, phi, t_jkl, r_kl):
    origin = k_coord
    a1 = origin - i_coord
    a2 = origin - j_coord
    r1 = R.from_rotvec(phi * a2/LA.norm(a2), degrees=True)
    n1 = np.cross(a2, a1)
    n2 = r1.apply(n1)
    m = np.cross(n2, a2)
    r2 = R.from_rotvec((t_jkl+90) * n2/LA.norm(n2), degrees=True)
    a3 = r2.apply(m)
    a3 = a3/LA.norm(a3)*r_kl
    
    l_coord = origin+a3
    return l_coord
    
def find_build_seq(topo_def, missing_atoms, missing_hydrogens):
    missing_atoms = list(missing_atoms)
    missing_hydrogens = list(missing_hydrogens)
    all_missing = missing_atoms + missing_hydrogens
    lookup_dict = topo_def.atom_lookup_dict
    running_dict = OrderedDict({k: lookup_dict[k] for k in all_missing})

    heavy_build_seq = recur_find_build_seq(
        running_dict,
        missing_atoms.copy(), # Prevent element removal on the original list
        build_seq = [], 
        exclude_list = missing_hydrogens
    )

    hydrogen_build_seq = recur_find_build_seq(
        running_dict,
        missing_hydrogens.copy(),
        build_seq = [],
        exclude_list = []
    )

    return heavy_build_seq, hydrogen_build_seq

def find_coords_by_ic(build_sequence, ic_dicts, coord_dict):
    computed_coords = []
    for atom_name, ic_key in build_sequence:
        i, j, k, l = ic_key
        ic_param_dict = ic_dicts[ic_key]
        is_improper = ic_param_dict['improper']
        phi = ic_param_dict['Phi']
        if atom_name == i:
            # the atom is i
            if is_improper:
                # i, j, *k, l => l, *k, j, i = a1, a2, a3, cur_atom
                a1, a2, a3 = coord_dict[j], coord_dict[l], coord_dict[k]
                bond_len = ic_param_dict['R(I-K)']
                bond_angle = ic_param_dict['T(I-K-J)']
                coord = get_coord_from_improper_ic(
                    a1, a2, a3, phi, bond_angle, bond_len
                )
            else:
                # i, j, k, l => l, k, j, i = a1, a2, a3, cur_atom
                a1, a2, a3 = coord_dict[l], coord_dict[k], coord_dict[j]
                bond_len = ic_param_dict['R(I-J)']
                bond_angle = ic_param_dict['T(I-J-K)']
                coord = get_coord_from_dihedral_ic(
                    a1, a2, a3, phi, bond_angle, bond_len
                )
        else:
            # the atom is l
            a1, a2, a3 = coord_dict[i], coord_dict[j], coord_dict[k]
            bond_len = ic_param_dict['R(K-L)']
            bond_angle = ic_param_dict['T(J-K-L)']
            if is_improper:
                coord = get_coord_from_improper_ic(
                    a1, a2, a3, phi, bond_angle, bond_len
                )
            else:
                coord = get_coord_from_dihedral_ic(
                    a1, a2, a3, phi, bond_angle, bond_len
                )
        computed_coords.append(atom_name)
        coord_dict[atom_name] = coord
        
    return computed_coords
    
def ab_initio_ic_build(topo_def):
    ic_dicts = topo_def.ic
    lookup_dict = topo_def.atom_lookup_dict
    all_atoms = [k for k in lookup_dict.keys() if k not in ('-C','CA','N')]
    running_dict = OrderedDict({k: lookup_dict[k] for k in all_atoms})

    build_seq = recur_find_build_seq(
        running_dict,
        all_atoms,
        build_seq = [],
        exclude_list = []
    )

    base_ic1 = ic_dicts[('-C', 'N', 'CA', 'C')]
    base_ic2 = ic_dicts[('N', 'CA', 'C', '+N')]
    r_CN = base_ic1['R(I-J)']
    t_CNCA = np.deg2rad(base_ic1['T(I-J-K)'])
    r_NCA = base_ic2['R(I-J)']

    x_N = np.cos(t_CNCA)*r_NCA
    y_N = np.sin(t_CNCA)*r_NCA

    coord_dict = {
        '-C': np.array([r_CN,0,0], dtype=float),
        'N': np.array([0, 0, 0], dtype=float),
        'CA':np.array([x_N, y_N, 0])
    }
    find_coords_by_ic(build_seq, ic_dicts, coord_dict)

    return coord_dict

class ResidueFixer:
    def __init__(self):
        self._res = None
        self.topo_def = None
        self.heavy_build_sequence = None
        self.hydrogen_build_sequence = None
        self.coord_dict = None
        self.res_type = None
        self.nei_atom_names = None

    @property
    def residue(self):
        """The residue to be fixed."""
        return self._res
    
    @residue.setter
    def residue(self, residue):
        self.load_residue(residue)

    def _create_empty_neighbor_atom_dict_from_ic(self):
        """Get the names of the neighboring atoms for the loaded residue."""
        nei_atoms = {}
        for ic_key in self.topo_def.ic:
            for atom in ic_key:
                if atom.startswith('+') or atom.startswith('-'):
                    nei_atoms[atom] = None
        return nei_atoms
    
    @staticmethod
    def _get_neighbor_atom_coord(atom_names, prev_nei, next_nei):
        """Get the coordinates of the neighboring atoms for list of atom names."""
        coord_dict = {}
        for atom_name in atom_names:
            if atom_name.startswith('+'):
                coord_dict[atom_name] = next_nei[atom_name[1:]].coord
            elif atom_name.startswith('-'):
                coord_dict[atom_name] = prev_nei[atom_name[1:]].coord
            else:
                raise ValueError(
                    'Atom name should start with "+" or "-".'
                )
        return coord_dict
    
    def _get_neighbor_backbone_coords(self):
        """Get the coordinates of the neighboring backbone atoms for the loaded 
        residue."""
        residue = self.residue
        chain = residue.parent
        
        if chain is None:
            warnings.warn(
                'Residue does not belong to a polymer chain, and no neighboring '
                'residues exist. The temperary backbone atoms of the neigbor '
                'residues will be calculated as place holders.'
            )
            return

        resseq = residue.id[1]
        list_idx = chain.child_list.index(residue)
        if (
            (list_idx == 0) or
            (list_idx == len(chain.child_list) - 1)
        ):
            if residue.topo_definition.patch_with is None:
                warnings.warn(
                    'Missing atoms on terminal residues will be built without patching! '
                    'Terminal patching is recommended before building missing atoms!'
                )
            return
        
        prev_res = chain.child_list[list_idx-1]
        next_res = chain.child_list[list_idx+1]
        rev_resseq, next_resseq = prev_res.id[1], next_res.id[1]
        if rev_resseq + 1 != resseq or next_resseq - 1 != resseq:
            warnings.warn(
                'Immediate neighbor residues missing! The temperary backbone atoms '
                'of the neigbor residues will be calculated as place holders.'
            )
            return
        
        neighbor_coord_dict = self._get_neighbor_atom_coord(
            self.nei_atom_names, prev_res, next_res
        )
        return neighbor_coord_dict
    
    def load_residue(self, residue, topo_definition = None):
        """Load the residue to be fixed."""
        if topo_definition is not None:
            self.topo_def = topo_definition
        elif residue.topo_definition is not None:
            self.topo_def = residue.topo_definition
        else:
            raise ValueError(
                'No ResidueDefinition provided nor loaded in the residue! '
                'Residue Topology Definition must be loaded in the residue using '
                'load_topo_definition() or provided as a parameter in the '
                'load_residue() method.'
            )
        self._res = residue
        if residue.resname in self.topo_def.aa_3to1:
            self.res_type = 'aa'
            self.nei_atom_names = ('-C', '+N', '+CA')
        elif residue.resname in ('A', 'G', 'C', 'U', 'T'):
            self.res_type = 'nuc'
            self.nei_atom_names = ("-O3'", "+P", "+O5'")
        else:
            raise ValueError(
                f'Residue {residue} has no neighbor backbone atoms to '
                'build! Due to unkown residue type'
                )
        self.coord_dict = {atom.name: atom.coord for atom in self._res}
        neighbor_coord_dict = self._get_neighbor_backbone_coords()
        if neighbor_coord_dict is None:
            # we only need the name of the missing atoms from neighbors, 
            # not to actually build them
            nei_atom_dict = self._create_empty_neighbor_atom_dict_from_ic()
            residue.missing_atoms.update(nei_atom_dict)
        else:
            self.coord_dict.update(neighbor_coord_dict)
        self.heavy_build_sequence, self.hydrogen_build_sequence = \
            find_build_seq(
            self.topo_def, self.missing_atoms, self.missing_hydrogens
        )

    @property
    def missing_atoms(self):
        """Get the current missing heavy atoms"""
        return self._res.missing_atoms

    @property
    def missing_hydrogens(self):
        """Get the current missing hydrogen atoms"""
        return self._res.missing_hydrogens

    def remove_hydrogens(self):
        """Remove all hydrogens on the residue"""
        hydrogens = []
        for atom in self._res:
            if atom.element == 'H':
                hydrogens.append(atom)
        for hydrogen in hydrogens:
            self._res.detach_child(hydrogen.id)
            if hydrogen in self._res.undefined_atoms:
                self._res.undefined_atoms.remove(hydrogen)
        # update missing hydrogens from the residue topology definition
        self._res.missing_hydrogens = {}
        for atom_def in self._res.topo_definition:
            if atom_def.element == 'H':
                self._res.missing_hydrogens[atom_def.name] = atom_def.create_new_atom()
        self.heavy_build_sequence, self.hydrogen_build_sequence = \
            find_build_seq(
            self.topo_def, self.missing_atoms, self.missing_hydrogens
        )


    def _build_atoms(self, build_sequence, missing_atoms:dict):
        computed_atom_names = find_coords_by_ic(
            build_sequence, self.topo_def.ic, self.coord_dict
        )
        built_atoms = []
        for atom_name in computed_atom_names:
            built_atom = missing_atoms.pop(atom_name)
            if atom_name.startswith('+') or atom_name.startswith('-'):
                # neighboring residue's backbone atoms will not be built
                continue
            coord = self.coord_dict[atom_name]
            built_atom.coord = coord
            self._res.add(built_atom)
            built_atoms.append(built_atom)
        return built_atoms
    
    def build_missing_atoms(self):
        """Build missing atoms based on residue topology definition on 
        internal coordinates. If neigbor residue has missing backbone atom that 
        the ic table depends on (namely +N, +CA, and -C), place holders for these 
        atoms will be built first."""
        if len(self.missing_atoms) == 0:
            return
        return self._build_atoms(self.heavy_build_sequence, self.missing_atoms)

    def build_hydrogens(self):
        """Build hydrogens atoms based on residue topology definition on 
        internal coordinates. Any missing heavy atoms will be built before building
        the hydrogens. If neigbor residue has missing backbone atom that the ic table
        depends on (namely +N, +CA, and -C), place holders for these atoms will be 
        built first."""

        if len(self.missing_hydrogens) == 0:
            return
        built_atoms = []
        if len(self.missing_atoms) != 0:
            missing_atom_names = tuple(self.missing_atoms.keys())
            if (
                missing_atom_names != ('-C', '+N', '+CA') or 
                missing_atom_names != ("-O3'", "+P", "+O5'")
            ):
                warnings.warn(
                    f'{len(self.missing_atoms)} Missing heavy atoms are built '
                    f'before building hydrogens: {missing_atom_names}'
                )
            built_atoms.extend(self.build_missing_atoms())
        built_atoms.extend(
            self._build_atoms(
                self.hydrogen_build_sequence, self.missing_hydrogens
            )
        )
        return built_atoms

    def remove_undefined_atoms(self):
        """Remove undefined atoms from the residue"""
        
        detach_ids = [atom.id for atom in self._res.undefined_atoms]
        for atom in self._res.undefined_atoms:
            for atom_group in self._res.atom_groups:
                if atom in atom_group:
                    atom_group.remove(atom)
        for atom_id in detach_ids:
            self._res.detach_child(atom_id)
        self._res.undefined_atoms = []

    def rebuild_hydrogens(self):
        """Rebuild hydrogens on the residue"""
        self.remove_hydrogens()
        built_hydrogens = self.build_hydrogens()
        return built_hydrogens

def build_missing_atoms_for_chain(chain):
    """Build missing residue for a PolymerChain where topology definitions 
    have been loaded for each residue. Missing atom will be built based on the IC
    definitions if the residue is not completely missing (e.g. missing loop on a
    chain).
    
    Args:
        chain: PolymerChain with topology definitions loaded for each residue.
    """
    built_atoms = {}
    res_builder = ResidueFixer()
    chain.sort_residues()
    for res in chain:
        if res.topo_definition is None:
            warnings.warn(f'No topology definition on {res}! Skipped')
            continue
        if res.missing_atoms:
            res_builder.load_residue(res)
            built_atoms[(res.id[1], res.resname)] = res_builder.build_missing_atoms()
    return built_atoms

def build_hydrogens_for_chain(chain, rebuild=False):
    """Build missing hydrogens for a PolymerChain where topology definitions 
    have been loaded for each residue. Missing atom will be built based on the IC
    definitions if the residue is not completely missing (e.g. missing loop on a
    chain).
    
    Args:
        chain: PolymerChain with topology definitions loaded for each residue.
        rebuild: If True, remove all hydrogens and rebuild them. If False, only
            build missing hydrogens.
    """
    built_atoms = {}
    res_builder = ResidueFixer()
    chain.sort_residues()
    for res in chain:
        if res.topo_definition is None:
            warnings.warn(f'No topology definition on {res}! Skipped')
            continue
        if not res.missing_hydrogens and not rebuild:
            continue
        res_builder.load_residue(res)
        if rebuild:
            res_builder.remove_hydrogens()
        built_atoms[(res.id[1], res.resname)] = res_builder.build_hydrogens()
    return built_atoms

def fix_chain(chain):
    """Fix a PolymerChain by building missing atoms and hydrogens based on 
    topology definitions. Missing atom will be built based on the IC definitions 
    if the residue is not completely missing (e.g. missing loop on a chain). Any 
    undefined atoms will be removed.
    
    Args:
        chain: PolymerChain with topology definitions loaded for each residue.
    
    Returns:
        A dictionary of built atoms with residue id and residue name as keys.
    """
    built_atoms = {}
    res_builder = ResidueFixer()
    chain.sort_residues()
    for res in chain:
        if res.topo_definition is None:
            warnings.warn(f'No topology definition on {res}! Skipped')
            continue
        if res.missing_atoms or res.missing_hydrogens:
            res_builder.load_residue(res)
            cur_built_atoms = built_atoms[(res.id[1], res.resname)] = []
            if built_heavy_atoms := res_builder.build_missing_atoms():
                cur_built_atoms.extend(built_heavy_atoms)
            # we force rebuild hydrogens if there are missing heavy atoms
            built_hydrogens = res_builder.rebuild_hydrogens()
            cur_built_atoms.extend(built_hydrogens)
        if res.undefined_atoms:
            res_builder.load_residue(res)
            res_builder.remove_undefined_atoms()
    return built_atoms