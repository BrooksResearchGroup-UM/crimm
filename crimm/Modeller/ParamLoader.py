import warnings
from crimm.IO.PRMParser import categorize_lines, parse_line_dict
from crimm.Modeller.TopoLoader import TopologyElementContainer
import crimm.StructEntities as Entities

class ParameterLoader:
    """A dictionary that stores parameters for CHARMM force field."""
    def __init__(self, file_path=None):
        self.param_dict = {}
        if file_path is not None:
            self.load_from_file(file_path)

    def load_from_file(self, filename):
        """Load parameters from a CHARMM prm file."""
        with open(filename, 'r', encoding='utf-8') as f:
            param_line_dict = categorize_lines(f.readlines())
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
                warnings.warn(f'No {topo_type} found.')
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
