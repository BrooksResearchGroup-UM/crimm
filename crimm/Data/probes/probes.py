import numpy as np
from Bio.Data.IUPACData import atom_weights
from crimm.StructEntities import Bond, Angle, Dihedral, Improper

bond_type_dict = {
    1: 'single',
    2: 'double',
    3: 'triple',
    4: 'quadruple',
    5: 'aromatic'
}

bond_order_dict = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 2
}

class _ProbeAtomDef:
    """Atom definition class for Probe molecules."""
    # This class is created for the purpose of compatibility with the
    # crimm.StructEntities.Atom class. It is not intended to be used
    # directly.
    def __init__(self, element, atom_type, charge) -> None:
        self.mass = atom_weights[element]
        self.atom_type = atom_type
        self.charge = charge
        # Donor and accetor flags are set by individual Probe class
        self.is_donor = False
        self.is_acceptor = False

    def __repr__(self):
        repr_str = f"<Probe Atom Definition type={self.atom_type}>"
        return repr_str

class ProbeAtom:
    """Atom class for Probe molecules."""
    altloc = ' '
    bfactor = 0.0
    occupancy = 1.0
    def __init__(self, name, coord, element,atom_type, charge):
        self.name = name
        self.fullname = name
        self.coord = np.asarray(coord, dtype=np.float64)
        self.element = element
        self.topo_definition = _ProbeAtomDef(element, atom_type, charge)
        self.serial_number = None
        self.parent = None

    def __repr__(self) -> str:
        return f'<Probe Atom {self.name}>'

    @property
    def id(self):
        return self.name

    def get_full_id(self):
        if self.parent is None:
            return self.id
        return self.parent.segid, self.parent.resname, self.id
    
    def get_serial_number(self):
        return self.serial_number

class _Probe:
    segid = 'PROB'
    level = 'R'
    child_list = []
    child_dict = {}
    parent = None
    _bonds = ()
    _bond_types = ()
    _angles = ()
    _dihedrals = ()
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id, resname):
        self.resname = resname
        self.res_id = res_id
        for i, atom in enumerate(self.child_list, start=1):
            atom.serial_number = i
            atom.parent = self
        self.H_donors = []
        for donor in self._donors:
            self.child_list[donor-1].topo_definition.is_donor = True
            self.H_donors.append(self.child_list[donor-1])
        self.H_acceptors = []
        for acceptor in self._acceptors:
            self.child_list[acceptor-1].topo_definition.is_acceptor = True
            self.H_acceptors.append(self.child_list[acceptor-1])
        self.bonds = self._assign_topo_elements(self._bonds, Bond)
        self._assign_bond_types()
        self.impropers = self._assign_topo_elements(self._impropers, Improper)
        self.angles = self._assign_topo_elements(self._angles, Angle)
        self.dihedrals = self._assign_topo_elements(self._dihedrals, Dihedral)
        self.child_dict = {atom.name: atom for atom in self.child_list}
        self._conformer_coords = np.empty((0, len(self.atoms), 3), dtype=np.float32)

    def __getitem__(self, key):
        return self.child_dict[key]

    def __iter__(self):
        return iter(self.child_list)

    def __len__(self):
        return len(self.child_list)
    
    def __repr__(self) -> str:
        return f'<Probe Molecule name={self.resname}>'

    def _assign_topo_elements(self, atom_serial_tuples, topo_element_type):
        elements = []
        for serial_tuple in atom_serial_tuples:
            atoms = [self.child_list[i-1] for i in serial_tuple]
            elements.append(topo_element_type(*atoms))
        return elements
    
    def _assign_bond_types(self):
        for bond, order in zip(self.bonds, self._bond_types):
            bond.order = bond_order_dict[order]
            bond.type = bond_type_dict[order]

    @property
    def id(self):
        return (' ', self.res_id, ' ')

    @property
    def atoms(self):
        return self.child_list
    
    @property
    def conformer_coords(self):
        """Return the coordinates of all conformers of the molecule. Returns an
        NxMx3 array, where N is the number of conformers and M is the number of
        atoms in the molecule."""
        return self._conformer_coords

    def get_atoms(self, include_alt=False):
        return iter(self.child_list)
    
    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from crimm.Visualization import show_nglview
        from IPython.display import display
        display(show_nglview(self))
        return repr(self)

    def add_conformer_coord(self, conformer_coord):
        """Add a conformer's coordinates to the molecule."""
        self._conformer_coords = np.append(
            self._conformer_coords, conformer_coord.reshape(1,-1,3), axis=0
        )

    def add_conformer_coords_multiple(self, conformer_coords):
        """Add multiple conformers' coordinates to the molecule. Accepts an
        NxMx3 array, where N is the number of conformers and M is the number of
        atoms in the molecule."""
        if conformer_coords.shape[1:] != (len(self.atoms), 3):
            raise ValueError(
                "Each conformer's coordinates must be Nx3 array, "
                "where N is the number of atoms in the molecule."
            )
        self._conformer_coords = np.append(
            self._conformer_coords, conformer_coords, axis=0
        )


class Acetaldehyde(_Probe):
    """Acetaldehyde probe for use in the PROBE grid generation and docking."""
    probe_id = 0
    probe_code = 'aald'
    child_list = [
        # name, coord, element, atom_type, charge, is_donor, is_acceptor
        ProbeAtom('HA', [-1.308, -1.047, -0.000], 'H', 'HGR52', 0.09),
        ProbeAtom('C', [-0.848, -0.036, 0.000], 'C', 'CG2O4', 0.20),
        ProbeAtom('O', [-1.580, 0.931, 0.000], 'O', 'OG2D1', -0.40),
        ProbeAtom('CB', [0.650, 0.023, -0.000], 'C', 'CG331', -0.16),
        ProbeAtom('HB1', [0.979, 1.085, -0.000], 'H', 'HGA3', 0.09),
        ProbeAtom('HB2', [1.054, -0.477, 0.906], 'H', 'HGA3', 0.09),
        ProbeAtom('HB3', [1.054, -0.477, -0.906], 'H', 'HGA3', 0.09),
    ]
    _bonds = ((1, 2), (2, 4), (4, 5), (4, 6), (4, 7), (2, 3))
    _bond_types = (1, 1, 1, 1, 1, 2)
    _angles = (
        (1, 2, 3), (1, 2, 4), (3, 2, 4), (2, 4, 5), (2, 4, 6),
        (2, 4, 7), (5, 4, 6), (5, 4, 7), (6, 4, 7)
    )
    _dihedrals = (
        (1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 4, 7),
        (3, 2, 4, 5), (3, 2, 4, 6), (3, 2, 4, 7)
    )
    _impropers = ((2, 4, 3, 1),)
    _donor = ()
    _acceptor = (3,)

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname='Acetaldehyde')


class Acetamide(_Probe):
    """Acetamide probe for use in the PROBE grid generation and docking."""
    probe_id = 1
    probe_code = 'acem'
    child_list = [
        # name, coord, element, atom_type, charge, is_donor, is_acceptor
        ProbeAtom('CC', [-1.083, -0.127, -0.000], 'C', 'HGR52', -0.27),
        ProbeAtom('C', [0.245, 0.537, 0.000], 'C', 'CG2O1', 0.55),
        ProbeAtom('N', [1.338, -0.261, 0.000], 'N', 'NG2S2', -0.62),
        ProbeAtom('HC', [2.222, 0.197, -0.000], 'H', 'HGP1', 0.32),
        ProbeAtom('HT', [1.231, -1.249, 0.000], 'H', 'HGP1', 0.3),
        ProbeAtom('O', [0.333, 1.760, 0.000], 'O', 'OG2D1', -0.55),
        ProbeAtom('HC1', [-1.876, 0.653, -0.000], 'H', 'HGA3', 0.09),
        ProbeAtom('HC2', [-1.204, -0.755, -0.907], 'H', 'HGA3', 0.09),
        ProbeAtom('HC3', [-1.205, -0.755, 0.907], 'H', 'HGA3', 0.09),
    ]
    _bonds = ((2, 3), (3, 4), (3, 5), (2, 6), (2, 1), (1, 7), (1, 8), (1, 9))
    _bond_types = (1, 1, 1, 2, 1, 1, 1, 1)
    _angles = (
        (2, 1, 7), (2, 1, 8), (2, 1, 9), (7, 1, 8), (7, 1, 9), (8, 1, 9),
        (1, 2, 3), (1, 2, 6), (3, 2, 6), (2, 3, 4), (2, 3, 5), (4, 3, 5)
    )
    _dihedrals = (
        (1, 2, 3, 4), (1, 2, 3, 5), (3, 2, 1, 7), (3, 2, 1, 8), (3, 2, 1, 9),
        (4, 3, 2, 6), (5, 3, 2, 6), (6, 2, 1, 7), (6, 2, 1, 8), (6, 2, 1, 9)
    )
    _impropers = ((2, 1, 3, 6),)
    _donor = ()
    _acceptor = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname='Acetamide')


class AceticAcid(_Probe):
    """Acetic acid probe for use in the PROBE grid generation and docking."""
    probe_id = 2
    probe_code = 'acet'
    child_list = [
        # name, coord, element, atom_type, charge, is_donor, is_acceptor
        ProbeAtom('C1', [0.646, 0.003, -0.000], 'C', 'CG331', -0.37),
        ProbeAtom('C2', [-0.862, 0.003, 0.000], 'C', 'CG2O3', 0.62),
        ProbeAtom('H1', [1.020, 1.046, 0.000], 'H', 'HGA3', 0.09),
        ProbeAtom('H2', [1.004, -0.525, -0.906], 'H', 'HGA3', 0.09),
        ProbeAtom('H3', [1.004, -0.525, 0.906], 'H', 'HGA3', 0.09),
        ProbeAtom('O1', [-1.412, 1.131, 0.000], 'O', 'OG2D2', -0.76),
        ProbeAtom('O2', [-1.398, -1.133, -0.000], 'O', 'OG2D2', -0.76),
    ]
    _bonds = ((1, 3), (1, 4), (1, 5), (1, 2), (2, 6), (2, 7))
    _bond_types = (1, 1, 1, 1, 2, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (2, 1, 5), (3, 1, 4), (3, 1, 5),
        (4, 1, 5), (1, 2, 6), (1, 2, 7), (6, 2, 7)
    )
    _dihedrals = (
        (3, 1, 2, 6), (3, 1, 2, 7), (4, 1, 2, 6), (4, 1, 2, 7),
        (5, 1, 2, 6), (5, 1, 2, 7)
    )
    _impropers = ((2, 7, 6, 1),)
    _donor = ()
    _acceptor = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname='Acetic Acid')


class Acetonitrile(_Probe):
    """Acetonitrile probe for use in the PROBE grid generation and docking."""
    probe_id = 3
    probe_code = 'acn'
    child_list = [
        # name, coord, element, atom_type, charge, is_donor, is_acceptor
        ProbeAtom('C1', [0.494, 0.000, 0.000], 'C', 'CG331', -0.17),
        ProbeAtom('H11', [0.876, 0.012, -1.042], 'H', 'HGA3', 0.09),
        ProbeAtom('H12', [0.876, -0.908, 0.510], 'H', 'HGA3', 0.09),
        ProbeAtom('H13', [0.876, 0.896, 0.531], 'H', 'HGA3', 0.09),
        ProbeAtom('C2', [-0.972, 0.000, 0.000], 'C', 'CG1N1', 0.36),
        ProbeAtom('N3', [-2.150, 0.000, 0.000], 'N', 'NG1T1', -0.46),
    ]
    _bonds = ((1, 5), (5, 6), (1, 2), (1, 3), (1, 4))
    _bond_types = (1, 3, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (2, 1, 5), (3, 1, 4), 
        (3, 1, 5), (4, 1, 5), (1, 5, 6)
    )
    _dihedrals = ()
    _impropers = ()
    _donor = ()
    _acceptor = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname='Acetonitrile')


class Acetone(_Probe):
    """Acetone probe for use in the PROBE grid generation and docking."""
    probe_id = 4
    probe_code = 'aco'
    child_list = [
        # name, coord, element, atom_type, charge, is_donor, is_acceptor
        ProbeAtom('O1', [0.000, -1.844, 0.000], 'O', 'OG2D3', -0.48),
        ProbeAtom('C1', [0.000, -0.619, 0.001], 'C', 'CG2O5', 0.40),
        ProbeAtom('C2', [-1.269, 0.165, 0.000], 'C', 'CG331', -0.23),
        ProbeAtom('C3', [1.268, 0.165, 0.000], 'C', 'CG331', -0.23),
        ProbeAtom('H21', [-2.127, -0.538, 0.007], 'H', 'HGA3', 0.09),
        ProbeAtom('H22', [-1.314, 0.808, 0.903], 'H', 'HGA3', 0.09),
        ProbeAtom('H23', [-1.321, 0.796, -0.911], 'H', 'HGA3', 0.09),
        ProbeAtom('H31', [2.127, -0.537, 0.050], 'H', 'HGA3', 0.09),
        ProbeAtom('H32', [1.340, 0.764, -0.931], 'H', 'HGA3', 0.09),
        ProbeAtom('H33', [1.296, 0.839, 0.881], 'H', 'HGA3', 0.09),
    ]
    _bonds = (
        (2, 3), (2, 4), (3, 5), (3, 6), (3, 7),
        (4, 8), (4, 9), (4, 10), (1, 2)
    )
    _bond_types = (1, 1, 1, 1, 1, 1, 1, 1, 2)
    _angles = (
        (1, 2, 3), (1, 2, 4), (3, 2, 4), (2, 3, 5), (2, 3, 6),
        (2, 3, 7), (5, 3, 6), (5, 3, 7), (6, 3, 7), (2, 4, 8),
        (2, 4, 9), (2, 4, 10),(8, 4, 9), (8, 4, 10),(9, 4, 10)
    )

    _dihedrals = (
        (1, 2, 3, 5), (1, 2, 3, 6), (1, 2, 3, 7), (1, 2, 4, 8),
        (1, 2, 4, 9), (1, 2, 4, 10),(3, 2, 4, 8), (3, 2, 4, 9),
        (3, 2, 4, 10),(4, 2, 3, 5), (4, 2, 3, 6), (4, 2, 3, 7)
    )
    _impropers = ((2, 3, 4, 1),)
    _donors = ()
    _acceptors = (1,)

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Acetone")


class Benzaldehyde(_Probe):
    """Benzaldehyde probe for use in the PROBE grid generation and docking."""
    probe_id = 5
    probe_code = "bald"
    child_list = (
        # name, coord, element, atom_type, charge
        ProbeAtom('HA', [-2.832, -1.022, 0.000], 'H', 'HGR52', 0.08),
        ProbeAtom('C', [-2.365, -0.015, 0.000], 'C', 'CG2O4', 0.24),
        ProbeAtom('O', [-3.101, 0.953, 0.000], 'O', 'OG2D1', -0.41),
        ProbeAtom('CG', [-0.878, 0.000, 0.000], 'C', 'CG2R61', 0.09),
        ProbeAtom('CD1', [-0.178, 1.217, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HD1', [-0.723, 2.151, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CE1', [1.224, 1.224, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HE1', [1.760, 2.163, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CZ', [1.929, 0.013, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HZ', [3.010, 0.019, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CD2', [-0.167, -1.211, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HD2', [-0.697, -2.152, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CE2', [1.235, -1.204, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HE2', [1.782, -2.136, 0.000], 'H', 'HGR61', 0.115),
    )
    _bonds = (
        (1, 2), (2, 4), (4, 5), (4, 11), (5, 6), (11, 12), (5, 7), 
        (11, 13), (7, 8), (13, 14), (7, 9), (13, 9), (9, 10), (2, 3)
    )
    _bond_types = (1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 1, 2)
    _angles = (
        (1, 2, 3), (1, 2, 4), (3, 2, 4), (2, 4, 5), (2, 4, 11), (5, 4, 11),
        (4, 5, 6), (4, 5, 7), (6, 5, 7), (5, 7, 8), (5, 7, 9), (8, 7, 9),
        (7, 9, 10), (7, 9, 13), (10, 9, 13), (4, 11, 12), (4, 11, 13),
        (12, 11, 13), (9, 13, 11), (9, 13, 14), (11, 13, 14)
    )
    _dihedrals = (
        (1, 2, 4, 5), (1, 2, 4, 11), (2, 4, 5, 6), (2, 4, 5, 7),
        (2, 4, 11, 12), (2, 4, 11, 13), (3, 2, 4, 5), (3, 2, 4, 11),
        (4, 5, 7, 8), (4, 5, 7, 9), (4, 11, 13, 9), (4, 11, 13, 14),
        (5, 4, 11, 12), (5, 4, 11, 13), (5, 7, 9, 10), (5, 7, 9, 13),
        (6, 5, 4, 11), (6, 5, 7, 8), (6, 5, 7, 9), (7, 5, 4, 11),
        (7, 9, 13, 11), (7, 9, 13, 14), (8, 7, 9, 10), (8, 7, 9, 13),
        (9, 13, 11, 12), (10, 9, 13, 11), (10, 9, 13, 14), (12, 11, 13, 14)
    )
    _impropers = ((2, 4, 3, 1),)
    _donors = ()
    _acceptors = (3,)

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Benzaldehyde")


class Benzene(_Probe):
    """Benzene probe for use in the PROBE grid generation and docking."""
    probe_id = 6
    probe_code = "benz"
    child_list = (
        # name, coord, element, atom_type, charge
        ProbeAtom('CG', [-1.395, 0.139, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HG', [-2.470, 0.246, -0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CD1', [-0.818, -1.138, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HD1', [-1.448, -2.016, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CD2', [-0.577, 1.277, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HD2', [-1.022, 2.262, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CE1', [0.577, -1.277, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HE1', [1.022, -2.262, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CE2', [0.818, 1.138, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HE2', [1.448, 2.016, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CZ', [1.395, -0.139, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HZ', [2.470, -0.246, -0.000], 'H', 'HGR61', 0.115),
    )

    _bonds = (
        (3, 1), (5, 1), (7, 3), (9, 5), (11, 7), (11, 9), (1, 2),
        (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)
    )
    _bond_types = (5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 5), (3, 1, 5), (1, 3, 4), (1, 3, 7),
        (4, 3, 7), (1, 5, 6), (1, 5, 9), (6, 5, 9), (3, 7, 8),
        (3, 7, 11), (8, 7, 11), (5, 9, 10), (5, 9, 11), (10, 9, 11),
        (7, 11, 9), (7, 11, 12), (9, 11, 12)
    )
    _dihedrals = (
        (1, 3, 7, 8), (1, 3, 7, 11), (1, 5, 9, 10), (1, 5, 9, 11),
        (2, 1, 3, 4), (2, 1, 3, 7), (2, 1, 5, 6), (2, 1, 5, 9),
        (3, 1, 5, 6), (3, 1, 5, 9), (3, 7, 11, 9), (3, 7, 11, 12),
        (4, 3, 1, 5), (4, 3, 7, 8), (4, 3, 7, 11), (5, 1, 3, 7),
        (5, 9, 11, 7), (5, 9, 11, 12), (6, 5, 9, 10), (6, 5, 9, 11),
        (7, 11, 9, 10), (8, 7, 11, 9), (8, 7, 11, 12), (10, 9, 11, 12)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Benzene")


class Cyclohexene(_Probe):
    """Cyclohexene probe for use in the PROBE grid generation and docking."""
    probe_id = 7
    probe_code = 'chxe'
    child_list = [
        # name, coord, element, atom_type, charge
        ProbeAtom('C1', [-1.502, 0.158, 0.099], 'C', 'CG321', -0.18),
        ProbeAtom('H11', [-2.394, 0.242, -0.56], 'H', 'HGA2', 0.09),
        ProbeAtom('H12', [-1.87, 0.007, 1.138], 'H', 'HGA2', 0.09),
        ProbeAtom('C2', [-0.699, -1.091, -0.318], 'C', 'CG321', -0.18),
        ProbeAtom('H21', [-0.578, -1.094, -1.425], 'H', 'HGA2', 0.09),
        ProbeAtom('H22', [-1.256, -2.01, 0.033], 'H', 'HGA2', 0.09),
        ProbeAtom('C3', [0.699, -1.091, 0.318], 'C', 'CG321', -0.18),
        ProbeAtom('H31', [1.256, -2.01, -0.033], 'H', 'HGA2', 0.09),
        ProbeAtom('H32', [0.578, -1.094, 1.425], 'H', 'HGA2', 0.09),
        ProbeAtom('C4', [1.502, 0.158, -0.099], 'C', 'CG321', -0.18),
        ProbeAtom('H41', [1.87, 0.007, -1.138], 'H', 'HGA2', 0.09),
        ProbeAtom('H42', [2.394, 0.242, 0.56], 'H', 'HGA2', 0.09),
        ProbeAtom('C5', [0.671, 1.417, -0.045], 'C', 'CG2D1', -0.15),
        ProbeAtom('H5', [1.218, 2.37, -0.101], 'H', 'HGA4', 0.15),
        ProbeAtom('C6', [-0.671, 1.417, 0.045], 'C', 'CG2D1', -0.15),
        ProbeAtom('H6', [-1.218, 2.37, 0.101], 'H', 'HGA4', 0.15),
    ]
    _bonds = (
        (1, 4), (4, 7), (7, 10), (10, 13), (13, 15), (15, 1),
        (15, 16), (1, 2), (1, 3), (4, 5), (4, 6), (7, 8), (7, 9),
        (10, 11), (10, 12), (13, 14)
    )
    _bond_types = (1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (2, 1, 15), (3, 1, 4), (3, 1, 15),
        (4, 1, 15), (1, 4, 5), (1, 4, 6), (1, 4, 7), (5, 4, 6),
        (5, 4, 7), (6, 4, 7), (4, 7, 8), (4, 7, 9), (4, 7, 10),
        (8, 7, 9), (8, 7, 10), (9, 7, 10), (7, 10, 11), (7, 10, 12),
        (7, 10, 13), (11, 10, 12), (11, 10, 13), (12, 10, 13), (10, 13, 14),
        (10, 13, 15), (14, 13, 15), (1, 15, 13), (1, 15, 16), (13, 15, 16)
    )
    _dihedrals = (
        (1, 4, 7, 8), (1, 4, 7, 9), (1, 4, 7, 10), (1, 15, 13, 10),
        (1, 15, 13, 14), (2, 1, 4, 5), (2, 1, 4, 6), (2, 1, 4, 7),
        (2, 1, 15, 13), (2, 1, 15, 16), (3, 1, 4, 5), (3, 1, 4, 6),
        (3, 1, 4, 7), (3, 1, 15, 13), (3, 1, 15, 16), (4, 1, 15, 13),
        (4, 1, 15, 16), (4, 7, 10, 11), (4, 7, 10, 12), (4, 7, 10, 13),
        (5, 4, 1, 15), (5, 4, 7, 8), (5, 4, 7, 9), (5, 4, 7, 10),
        (6, 4, 1, 15), (6, 4, 7, 8), (6, 4, 7, 9), (6, 4, 7, 10),
        (7, 4, 1, 15), (7, 10, 13, 14), (7, 10, 13, 15), (8, 7, 10, 11),
        (8, 7, 10, 12), (8, 7, 10, 13), (9, 7, 10, 11), (9, 7, 10, 12),
        (9, 7, 10, 13), (10, 13, 15, 16), (11, 10, 13, 14), (11, 10, 13, 15),
        (12, 10, 13, 14), (12, 10, 13, 15), (14, 13, 15, 16)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Cyclohexene")


class DimethylEther(_Probe):
    """Dimethyl probe for use in the PROBE grid generation and docking."""
    probe_id = 8
    probe_code = "dmee"
    child_list = [
        # name, coord, element, atom_type, charge
        ProbeAtom("C1", [-1.168, -0.000, -0.050], "C", "CG331", -0.10),
        ProbeAtom("H11", [-2.065, -0.000, -0.706], "H", "HGA3", 0.09),
        ProbeAtom("H12", [-1.194, -0.907, 0.591], "H", "HGA3", 0.09),
        ProbeAtom("H13", [-1.194, 0.907, 0.591], "H", "HGA3", 0.09),
        ProbeAtom("O2", [0.000, -0.000, -0.854], "O", "OG301", -0.34),
        ProbeAtom("C3", [1.168, -0.000, -0.050], "C", "CG331", -0.10),
        ProbeAtom("H31", [2.065, -0.000, -0.706], "H", "HGA3", 0.09),
        ProbeAtom("H32", [1.194, 0.907, 0.591], "H", "HGA3", 0.09),
        ProbeAtom("H33", [1.194, -0.907, 0.591], "H", "HGA3", 0.09),
    ]
    _bonds = ((1, 5), (5, 6), (1, 2), (1, 3), (1, 4), (6, 7), (6, 8), (6, 9))
    _bond_types = (1, 1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (2, 1, 5), (3, 1, 4), (3, 1, 5),
        (4, 1, 5), (1, 5, 6), (5, 6, 7), (5, 6, 8), (5, 6, 9),
        (7, 6, 8), (7, 6, 9), (8, 6, 9)
    )
    _dihedrals = (
        (1, 5, 6, 7), (1, 5, 6, 8), (1, 5, 6, 9),
        (2, 1, 5, 6), (3, 1, 5, 6), (4, 1, 5, 6)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="DimethylEther")


class Dimethylformamide(_Probe):
    """Dimethylformamide probe for use in the PROBE grid generation and docking."""
    probe_id = 9
    probe_code = "dmf"
    child_list = [
        # name, coord, element, atom_type, charge
        ProbeAtom('HA', [0.850, -2.053, 0.000], 'H', 'HGR52', 0.080),
        ProbeAtom('C', [-0.112, -1.520, 0.000], 'C', 'CG2O1', 0.430),
        ProbeAtom('O', [-1.180, -2.130, 0.000], 'O', 'OG2D1', -0.540),
        ProbeAtom('N', [-0.013, -0.159, 0.000], 'N', 'NG2S0', -0.330),
        ProbeAtom('CC', [-1.191, 0.689, -0.000], 'C', 'CG331', -0.090),
        ProbeAtom('HC1', [-2.121, 0.076, 0.000], 'H', 'HGA3', 0.090),
        ProbeAtom('HC2', [-1.145, 1.318, -0.914], 'H', 'HGA3', 0.090),
        ProbeAtom('HC3', [-1.145, 1.318, 0.914], 'H', 'HGA3', 0.090),
        ProbeAtom('CT', [1.284, 0.490, 0.000], 'C', 'CG331', -0.090),
        ProbeAtom('HT1', [2.100, -0.267, 0.000], 'H', 'HGA3', 0.090),
        ProbeAtom('HT2', [1.336, 1.120, 0.914], 'H', 'HGA3', 0.090),
        ProbeAtom('HT3', [1.336, 1.120, -0.914], 'H', 'HGA3', 0.090),
    ]
    _bonds = (
        (10, 9), (11, 9), (12, 9), (2, 1), (2, 4),
        (4, 5), (4, 9), (6, 5), (7, 5), (8, 5), (2, 3)
    )
    _bond_types = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)
    _angles = (
        (1, 2, 3), (1, 2, 4), (3, 2, 4), (2, 4, 5), (2, 4, 9),
        (5, 4, 9), (4, 5, 6), (4, 5, 7), (4, 5, 8), (6, 5, 7),
        (6, 5, 8), (7, 5, 8), (4, 9, 10), (4, 9, 11), (4, 9, 12),
        (10, 9, 11), (10, 9, 12), (11, 9, 12)
    )
    _dihedrals = (
        (1, 2, 4, 5), (1, 2, 4, 9), (2, 4, 5, 6), (2, 4, 5, 7), (2, 4, 5, 8),
        (2, 4, 9, 10), (2, 4, 9, 11), (2, 4, 9, 12), (3, 2, 4, 5), (3, 2, 4, 9),
        (5, 4, 9, 10), (5, 4, 9, 11), (5, 4, 9, 12), (6, 5, 4, 9), (7, 5, 4, 9),
        (8, 5, 4, 9)
    )
    _impropers = ((2, 4, 3, 1),)
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Dimethylformamide")


class Ethane(_Probe):
    """Ethane probe for use in the PROBE grid generation and docking."""
    probe_id = 10
    probe_code = 'etha'
    child_list = [
        ProbeAtom('H11', [-1.157, -0.402, -0.960], 'H', 'HGA3', 0.090),
        ProbeAtom('H12', [-1.157, -0.630, 0.828], 'H', 'HGA3', 0.090),
        ProbeAtom('H13', [-1.157, 1.032, 0.132], 'H', 'HGA3', 0.090),
        ProbeAtom('C1', [-0.766, -0.000, 0.000], 'C', 'CG331', -0.270),
        ProbeAtom('H21', [1.157, -1.032, -0.131], 'H', 'HGA3', 0.090),
        ProbeAtom('H22', [1.157, 0.630, -0.829], 'H', 'HGA3', 0.090),
        ProbeAtom('H23', [1.157, 0.403, 0.960], 'H', 'HGA3', 0.090),
        ProbeAtom('C2', [0.766, -0.000, 0.000], 'C', 'CG331', -0.270),
    ]
    _bonds = ((4, 1), (4, 2), (4, 3), (4, 8), (8, 5), (8, 6), (8, 7))
    _bond_types = (1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (1, 4, 2), (1, 4, 3), (1, 4, 8), (2, 4, 3), (2, 4, 8), (3, 4, 8),
        (4, 8, 5), (4, 8, 6), (4, 8, 7), (5, 8, 6), (5, 8, 7), (6, 8, 7)
    )
    _dihedrals = (
        (1, 4, 8, 5), (1, 4, 8, 6), (1, 4, 8, 7), (2, 4, 8, 5),
        (2, 4, 8, 6), (2, 4, 8, 7), (3, 4, 8, 5), (3, 4, 8, 6),
        (3, 4, 8, 7)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Ethane")


class Ethanol(_Probe):
    """Ethanol probe for use in the PROBE grid generation and docking."""
    probe_id = 11
    probe_code = 'etoh'
    child_list = [
        ProbeAtom('C1', [-0.432, -0.337, -0.000], 'C', 'CG321', 0.050),
        ProbeAtom('O1', [-1.276, 0.805, -0.000], 'O', 'OG311', -0.650),
        ProbeAtom('HO1', [-2.186, 0.496, -0.000], 'H', 'HGP1', 0.420),
        ProbeAtom('H11', [-0.633, -0.953, 0.906], 'H', 'HGA2', 0.090),
        ProbeAtom('H12', [-0.633, -0.953, -0.906], 'H', 'HGA2', 0.090),
        ProbeAtom('C2', [1.014, 0.141, -0.000], 'C', 'CG331', -0.270),
        ProbeAtom('H21', [1.713, -0.721, 0.000], 'H', 'HGA3', 0.090),
        ProbeAtom('H22', [1.216, 0.761, 0.900], 'H', 'HGA3', 0.090),
        ProbeAtom('H23', [1.216, 0.761, -0.900], 'H', 'HGA3', 0.090)
    ]
    _bonds = ((1, 6), (1, 2), (1, 4), (1, 5), (2, 3), (6, 7), (6, 8), (6, 9))
    _bond_types = (1, 1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 4), (2, 1, 5), (2, 1, 6), (4, 1, 5), (4, 1, 6), (5, 1, 6),
        (1, 2, 3), (1, 6, 7), (1, 6, 8), (1, 6, 9), (7, 6, 8), (7, 6, 9),
        (8, 6, 9)
    )
    _dihedrals = (
        (2, 1, 6, 7), (2, 1, 6, 8), (2, 1, 6, 9), (3, 2, 1, 4), (3, 2, 1, 5),
        (3, 2, 1, 6), (4, 1, 6, 7), (4, 1, 6, 8), (4, 1, 6, 9), (5, 1, 6, 7),
        (5, 1, 6, 8), (5, 1, 6, 9)
    )
    _impropers = ()
    _donors = (3,)
    _acceptors = (2,)

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Ethanol")


class Methylamine(_Probe):
    """Methylamine probe for use in the PROBE grid generation and docking."""
    probe_id = 12
    probe_code = 'mam1'
    child_list = [
        ProbeAtom('N1', [0.854, -0.000, -0.301], 'N', 'NG321', -0.990),
        ProbeAtom('C1', [-0.609, 0.000, -0.006], 'C', 'CG3AM2', -0.060),
        ProbeAtom('HN1', [1.354, -0.811, 0.056], 'H', 'HGPAM2', 0.390),
        ProbeAtom('HN2', [1.354, 0.811, 0.056], 'H', 'HGPAM2', 0.390),
        ProbeAtom('HC1', [-1.082, 0.882, -0.437], 'H', 'HGAAM2', 0.090),
        ProbeAtom('HC2', [-0.789, 0.000, 1.069], 'H', 'HGAAM2', 0.090),
        ProbeAtom('HC3', [-1.082, -0.882, -0.437], 'H', 'HGAAM2', 0.090)
    ]
    _bonds = ((1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7))
    _bond_types = (1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (3, 1, 4), (1, 2, 5), (1, 2, 6),
        (1, 2, 7), (5, 2, 6), (5, 2, 7), (6, 2, 7)
    )
    _dihedrals = (
        (3, 1, 2, 5), (3, 1, 2, 6), (3, 1, 2, 7), (4, 1, 2, 5),
        (4, 1, 2, 6), (4, 1, 2, 7)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Methylamine")


class Methylammonium(_Probe):
    """Methylammonium probe for use in the PROBE grid generation and docking."""
    probe_id = 13
    probe_code = 'mamm'
    child_list = [
        ProbeAtom('CE', [-0.741, 0.000, 0.000], 'C', 'CG334', 0.040),
        ProbeAtom('NZ', [0.746, -0.000, 0.000], 'N', 'NG3P3', -0.30),
        ProbeAtom('HE1', [-1.112, -1.028, -0.206], 'H', 'HGA3', 0.090),
        ProbeAtom('HE2', [-1.112, 0.692, -0.787], 'H', 'HGA3', 0.090),
        ProbeAtom('HE3', [-1.112, 0.336, 0.993], 'H', 'HGA3', 0.090),
        ProbeAtom('HZ1', [1.110, -0.645, 0.734], 'H', 'HGP2', 0.033),
        ProbeAtom('HZ2', [1.110, 0.958, 0.192], 'H', 'HGP2', 0.033),
        ProbeAtom('HZ3', [1.110, -0.313, -0.925], 'H', 'HGP2', 0.033)
    ]
    _bonds = ((1, 3), (1, 4), (1, 5), (1, 2), (2, 6), (2, 7), (2, 8))
    _bond_types = (1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (2, 1, 5), (3, 1, 4), (3, 1, 5),
        (4, 1, 5), (1, 2, 6), (1, 2, 7), (1, 2, 8), (6, 2, 7),
        (6, 2, 8), (7, 2, 8)
    )
    _dihedrals = (
        (3, 1, 2, 6), (3, 1, 2, 7), (3, 1, 2, 8), (4, 1, 2, 6),
        (4, 1, 2, 7), (4, 1, 2, 8), (5, 1, 2, 6), (5, 1, 2, 7),
        (5, 1, 2, 8)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Methylammonium")


class Phenol(_Probe):
    """Phenol probe for use in the PROBE grid generation and docking."""
    probe_id = 14
    probe_code = 'phen'
    child_list = [
        ProbeAtom('CG', [-1.635, 0.327, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HG', [-2.685, 0.582, -0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CD1', [-1.241, -1.019, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HD1', [-1.986, -1.801, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CD2', [-0.667, 1.341, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HD2', [-0.972, 2.378, -0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CE1', [0.122, -1.348, 0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HE1', [0.431, -2.383, 0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CE2', [0.696, 1.009, -0.000], 'C', 'CG2R61', -0.115),
        ProbeAtom('HE2', [1.436, 1.794, -0.000], 'H', 'HGR61', 0.115),
        ProbeAtom('CZ', [1.088, -0.336, 0.000], 'C', 'CG2R61', 0.110),
        ProbeAtom('OH', [2.455, -0.679, 0.000], 'O', 'OG311', -0.530),
        ProbeAtom('HH', [2.958, 0.136, -0.000], 'H', 'HGP1', 0.420)
    ]
    _bonds = (
        (5, 1), (7, 3), (11, 9), (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (12, 13), (3, 1), (9, 5), (11, 7)
    )
    _bond_types = (5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5)
    _angles = (
        (2, 1, 3), (2, 1, 5), (3, 1, 5), (1, 3, 4), (1, 3, 7),
        (4, 3, 7), (1, 5, 6), (1, 5, 9), (6, 5, 9), (3, 7, 8),
        (3, 7, 11), (8, 7, 11), (5, 9, 10), (5, 9, 11), (10, 9, 11),
        (7, 11, 9), (7, 11, 12), (9, 11, 12), (11, 12, 13)
    )
    _dihedrals = (
        (1, 3, 7, 8), (1, 3, 7, 11), (1, 5, 9, 10), (1, 5, 9, 11),
        (2, 1, 3, 4), (2, 1, 3, 7), (2, 1, 5, 6), (2, 1, 5, 9),
        (3, 1, 5, 6), (3, 1, 5, 9), (3, 7, 11, 9), (3, 7, 11, 12),
        (4, 3, 1, 5), (4, 3, 7, 8), (4, 3, 7, 11), (5, 1, 3, 7),
        (5, 9, 11, 7), (5, 9, 11, 12), (6, 5, 9, 10), (6, 5, 9, 11),
        (7, 11, 9, 10), (7, 11, 12, 13), (8, 7, 11, 9), (8, 7, 11, 12),
        (9, 11, 12, 13), (10, 9, 11, 12)
    )
    _impropers = ()
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Phenol")


class Isopropanol(_Probe):
    """Isopropanol probe for use in the PROBE grid generation and docking."""
    probe_id = 15
    probe_code = "pro2"
    child_list = [
        ProbeAtom('C2', [-0.002, -0.366, 0.288], 'C', 'CG311', 0.14),
        ProbeAtom('O2', [-0.232, -1.529, -0.48], 'O', 'OG311', -0.65),
        ProbeAtom('HO2', [-1.089, -1.861, -0.205], 'H', 'HGP1', 0.42),
        ProbeAtom('H21', [-0.032, -0.626, 1.373], 'H', 'HGA1', 0.09),
        ProbeAtom('C1', [-1.104, 0.64, -0.04], 'C', 'CG331', -0.27),
        ProbeAtom('H11', [-2.104, 0.236, 0.223], 'H', 'HGA3', 0.09),
        ProbeAtom('H12', [-1.092, 0.879, -1.126], 'H', 'HGA3', 0.09),
        ProbeAtom('H13', [-0.944, 1.581, 0.526], 'H', 'HGA3', 0.09),
        ProbeAtom('C3', [1.387, 0.163, -0.07], 'C', 'CG331', -0.27),
        ProbeAtom('H31', [2.161, -0.601, 0.155], 'H', 'HGA3', 0.09),
        ProbeAtom('H32', [1.614, 1.082, 0.508], 'H', 'HGA3', 0.09),
        ProbeAtom('H33', [1.437, 0.402, -1.154], 'H', 'HGA3', 0.09)
    ]
    _bonds = (
        (5, 1), (1, 9), (1, 2), (1, 4), (2, 3), (5, 6),
        (5, 7), (5, 8), (9, 10), (9, 11), (9, 12)
    )
    _bond_types = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 4), (2, 1, 5), (2, 1, 9), (4, 1, 5), (4, 1, 9), (5, 1, 9),
        (1, 2, 3), (1, 5, 6), (1, 5, 7), (1, 5, 8), (6, 5, 7), (6, 5, 8),
        (7, 5, 8), (1, 9, 10), (1, 9, 11), (1, 9, 12), (10, 9, 11), (10, 9, 12),
        (11, 9, 12)
    )
    _dihedrals = (
        (2, 1, 5, 6), (2, 1, 5, 7), (2, 1, 5, 8), (2, 1, 9, 10), (2, 1, 9, 11),
        (2, 1, 9, 12), (3, 2, 1, 4), (3, 2, 1, 5), (3, 2, 1, 9), (4, 1, 5, 6),
        (4, 1, 5, 7), (4, 1, 5, 8), (4, 1, 9, 10), (4, 1, 9, 11), (4, 1, 9, 12),
        (5, 1, 9, 10), (5, 1, 9, 11), (5, 1, 9, 12), (6, 5, 1, 9), (7, 5, 1, 9),
        (8, 5, 1, 9)
    )
    _impropers = ()
    _donors = (2,)
    _acceptors = (2,)

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Isopropanol")


class TertButanol(_Probe):
    """tert-Butanol probe for use in the PROBE grid generation and docking."""
    probe_id = 16
    probe_code = "tboh"
    child_list = [
        ProbeAtom('C', [0.017, 0.000, -0.240], 'C', 'CG301', 0.23),
        ProbeAtom('O', [-0.219, 0.000, -1.625], 'O', 'OG311', -0.65),
        ProbeAtom('HO', [-1.173, -0.000, -1.699], 'H', 'HGP1', 0.42),
        ProbeAtom('C1', [-0.618, -1.263, 0.333], 'C', 'CG331', -0.27),
        ProbeAtom('H11', [-0.177, -2.165, -0.143], 'H', 'HGA3', 0.09),
        ProbeAtom('H12', [-1.714, -1.269, 0.162], 'H', 'HGA3', 0.09),
        ProbeAtom('H13', [-0.432, -1.318, 1.426], 'H', 'HGA3', 0.09),
        ProbeAtom('C2', [-0.619, 1.262, 0.333], 'C', 'CG331', -0.27),
        ProbeAtom('H21', [-0.179, 2.165, -0.142], 'H', 'HGA3', 0.09),
        ProbeAtom('H22', [-0.433, 1.317, 1.426], 'H', 'HGA3', 0.09),
        ProbeAtom('H23', [-1.715, 1.268, 0.162], 'H', 'HGA3', 0.09),
        ProbeAtom('C3', [1.529, 0.000, -0.036], 'C', 'CG331', -0.27),
        ProbeAtom('H31', [1.980, 0.901, -0.503], 'H', 'HGA3', 0.09),
        ProbeAtom('H32', [1.981, -0.900, -0.503], 'H', 'HGA3', 0.09),
        ProbeAtom('H33', [1.772, 0.000, 1.047], 'H', 'HGA3', 0.09)
    ]
    _bonds = (
        (4, 1), (8, 1), (12, 1), (1, 2), (2, 3), (4, 5), (4, 6),
        (4, 7), (8, 9), (8, 10), (8, 11), (12, 13), (12, 14), (12, 15)
    )
    _bond_types = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 4), (2, 1, 8), (2, 1, 12), (4, 1, 8), (4, 1, 12),
        (8, 1, 12), (1, 2, 3), (1, 4, 5), (1, 4, 6), (1, 4, 7),
        (5, 4, 6), (5, 4, 7), (6, 4, 7), (1, 8, 9), (1, 8, 10),
        (1, 8, 11), (9, 8, 10), (9, 8, 11), (10, 8, 11), (1, 12, 13),
        (1, 12, 14), (1, 12, 15), (13, 12, 14), (13, 12, 15), (14, 12, 15)
    )
    _dihedrals = (
        (2, 1, 4, 5), (2, 1, 4, 6), (2, 1, 4, 7), (2, 1, 8, 9),
        (2, 1, 8, 10), (2, 1, 8, 11), (2, 1, 12, 13), (2, 1, 12, 14),
        (2, 1, 12, 15), (3, 2, 1, 4), (3, 2, 1, 8), (3, 2, 1, 12),
        (4, 1, 8, 9), (4, 1, 8, 10), (4, 1, 8, 11), (4, 1, 12, 13),
        (4, 1, 12, 14), (4, 1, 12, 15), (5, 4, 1, 8), (5, 4, 1, 12),
        (6, 4, 1, 8), (6, 4, 1, 12), (7, 4, 1, 8), (7, 4, 1, 12),
        (8, 1, 12, 13), (8, 1, 12, 14), (8, 1, 12, 15), (9, 8, 1, 12),
        (10, 8, 1, 12), (11, 8, 1, 12)
    )
    _impropers = ()
    _donors = (2,)
    _acceptors = (2,)

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="TertButanol")


class Urea(_Probe):
    """Urea probe for use in the PROBE grid generation and docking."""
    probe_id = 17
    probe_code = "urea"
    child_list = [
        ProbeAtom('N1', [-1.105, 0.218, 0.000], 'N', 'NG2S2', -0.69),
        ProbeAtom('H11', [-1.962, -0.284, 0.000], 'H', 'HGP1', 0.34),
        ProbeAtom('H12', [-1.036, 1.205, 0.000], 'H', 'HGP1', 0.34),
        ProbeAtom('C2', [0.000, -0.525, 0.000], 'C', 'CG2O6', 0.6),
        ProbeAtom('O2', [0.000, -1.752, 0.000], 'O', 'OG2D1', -0.58),
        ProbeAtom('N3', [1.105, 0.218, 0.000], 'N', 'NG2S2', -0.69),
        ProbeAtom('H31', [1.962, -0.284, 0.000], 'H', 'HGP1', 0.34),
        ProbeAtom('H32', [1.036, 1.205, 0.000], 'H', 'HGP1', 0.34)
    ]
    _bonds = ((4, 5), (4, 1), (4, 6), (1, 2), (1, 3), (6, 7), (6, 8))
    _bond_types = (2, 1, 1, 1, 1, 1, 1)
    _angles = (
        (2, 1, 3), (2, 1, 4), (3, 1, 4), (1, 4, 5), (1, 4, 6),
        (5, 4, 6), (4, 6, 7), (4, 6, 8), (7, 6, 8)
    )
    _dihedrals = (
        (1, 4, 6, 7), (1, 4, 6, 8), (2, 1, 4, 5), (2, 1, 4, 6),
        (3, 1, 4, 5), (3, 1, 4, 6), (5, 4, 6, 7), (5, 4, 6, 8)
    )
    _impropers = ((4, 1, 5, 6),)
    _donors = ()
    _acceptors = ()

    def __init__(self, res_id = 0):
        super().__init__(res_id, resname="Urea")

def create_new_probe_set():
    """Creates dictionary of probes for use in the PROBE grid generation and docking."""
    probe_set = {
        "acetaldehyde": Acetaldehyde(),
        "acetamide": Acetamide(),
        "acetic_acid": AceticAcid(),
        "acetonitrile": Acetonitrile(),
        "acetone": Acetone(),
        "benzaldehyde": Benzaldehyde(),
        "benzene": Benzene(),
        "cyclohexene": Cyclohexene(),
        "dimethyl_ether": DimethylEther(),
        "dimethylformamide": Dimethylformamide(),
        "ethane": Ethane(),
        "ethanol": Ethanol(),
        "methylamine": Methylamine(),
        "methylammonium": Methylammonium(),
        "phenol": Phenol(),
        "isopropanol": Isopropanol(),
        "tertbutanol": TertButanol(),
        "urea": Urea()
    }
    return probe_set
