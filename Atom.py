import warnings
from Bio.PDB.Atom import Atom as _Atom
from Bio.PDB.Atom import DisorderedAtom as _DisorderedAtom

class Atom(_Atom):
    """Atom class derived from Biopython Residue and made compatible with
    CHARMM"""
    def __init__(
        self,
        name,
        coord,
        bfactor,
        occupancy,
        altloc,
        fullname,
        serial_number,
        element=None,
        pqr_charge=None,
        radius=None,
    ) -> None:
        self.level = "A"
        # Reference to the residue
        self.parent = None
        # the atomic data
        self.name = name  # eg. CA, spaces are removed from atom name
        self.fullname = fullname  # e.g. " CA ", spaces included
        self.coord = coord
        self.bfactor = bfactor
        self.occupancy = occupancy
        self.altloc = altloc
        self.full_id = None  # (structure id, model id, chain id, residue id, atom id)
        self.id = name  # id of atom is the atom name (e.g. "CA")
        self.disordered_flag = 0
        self.anisou_array = None
        self.siguij_array = None
        self.sigatm_array = None
        self.orig_serial_number = serial_number
        self.serial_number = serial_number
        # Dictionary that keeps additional properties
        self.xtra = {}
        assert not element or element == element.upper(), element
        self.element = self._assign_element(element)
        self.mass = self._assign_atom_mass()
        self.pqr_charge = pqr_charge
        self.radius = radius

        # For atom sorting (protein backbone atoms first)
        self._sorting_keys = {"N": 0, "CA": 1, "C": 2, "O": 3}
        # Forcefield Parameters and Topology Definitions
        self._topo_def = None
        
    def reset_atom_serial_number(self):
        """Reset all atom serial numbers in the entire structure starting from 1."""
        top_parent = self.get_top_parent()
        if top_parent is self:
            self.serial_number = 1
        else:
            top_parent._reset_atom_serial_numbers()

    @property
    def topo_definition(self):
        """Topology related parameters for the atom"""
        return self._topo_def

    @topo_definition.setter
    def topo_definition(self, atom_def):
        if atom_def.name != self.name:
            warnings.warn(
                f"Atom Name Mismatch: Definition={atom_def.name} Atom={self.name}"
            )
        self._topo_def = atom_def
        self.id = self._topo_def.name

    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent.get_top_parent()
    
class DisorderedAtom(_DisorderedAtom):
    """Disoreded Atom class derived from Biopython Disordered Atom and made compatible with
    OpenMM Atom."""
    def __init__(self, id):
        super().__init__(id)

    def _find_top_parent(self):
        if self.parent is None:
            return self
        return self.parent._find_top_parent()
