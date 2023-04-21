import os
import warnings
from crimm.IO.PRMParser import categorize_lines, parse_line_dict, skip_line
from crimm.Modeller.TopoLoader import TopologyElementContainer
import crimm.StructEntities as Entities

toppar_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../Data/toppar')
)
prm_path_dict = {
    "protein": os.path.join(toppar_dir, 'prot.prm'),
    "nucleic": os.path.join(toppar_dir, 'na.prm'),
    "lipid": os.path.join(toppar_dir, 'lipid.prm'),
    "carb": os.path.join(toppar_dir, 'carb.prm'),
    "ethers": os.path.join(toppar_dir, 'ethers.prm'),
    "cgenff": os.path.join(toppar_dir, 'cgenff.prm')
}


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
    def __init__(self, file_path=None):
        self.param_dict = {}
        self._raw_data_strings = []
        if file_path is not None:
            self.load_type(file_path)

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
    
    def apply(self, topo_element_container: TopologyElementContainer):
        """Apply the parameter for a list of topology element"""
        if not isinstance(topo_element_container, TopologyElementContainer):
            raise TypeError(
                'Invalid argument type provided! TopologyElementContainer'
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
    
    def res_def_fill_ic(self, residue_definition, preserve = True):
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
                ic_table[ic_type] = self._find_ic_param(cur_ic_atom_types)

    def _find_ic_param(self, key):
        if len(key) == 2:
            bond_param = self.get_bond(key)
            return bond_param.b0
        else:
            angle_param = self.get_angle(key)
            return angle_param.theta0

    def fill_ic(self, topology_loader, preserve = True):
        """Fill in the missing parameters for the internal coordinates table
        of a topology."""
        for residue_definition in topology_loader.residues:
            self.res_def_fill_ic(residue_definition, preserve)
        # Also fill in the patched residue definitions
        for residue_definition in topology_loader.patched_defs.values():
            self.res_def_fill_ic(residue_definition, preserve)