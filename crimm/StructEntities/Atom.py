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
        topo_definition = None
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
        # Original serial number
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
        # For atom sorting (nucleic acid backbone atoms first)
        self._sorting_keys.update(
              {
                  "P": 0, "OP1": 1, "OP2": 2, "O5'": 3, 
                  "C5'": 4, "C4'": 5, "C3'": 6, "O3'": 7
              }
        )
        # For neighbor lookup and graph building
        self.neighbors = set()
        # Forcefield Parameters and Topology Definitions
        self._topo_def = None
        if topo_definition is not None:
            self.topo_definition = topo_definition

    def __repr__(self):
        """Print Atom object as <Atom atom_name>. if coord is None, print as
        <MissingAtom atom_name>"""
        if self.coord is None:
            return f"<MissingAtom {self.get_id()}>"
        return f"<Atom {self.get_id()}>"

    def __getstate__(self):
        """Return state of the atom object for pickling, excluding neighbors 
        to avoid infinite recursion errors"""
        state = {k: v for k, v in self.__dict__.items() if k != "neighbors"}
        return state

    def __setstate__(self, state):
        """Set state of the atom object for pickling"""
        self.__dict__.update(state)

    def reset_atom_serial_numbers(self):
        """Reset all atom serial numbers in the entire structure starting from 1."""
        top_parent = self.get_top_parent()
        if top_parent is self:
            self.set_serial_number(1)
        else:
            top_parent.reset_atom_serial_numbers()

    @property
    def topo_definition(self):
        """Topology related definitions and parameters for the atom"""
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
        """Get the highest level of encompassing entity"""
        if self.parent is None:
            return self
        return self.parent.get_top_parent()
    
class DisorderedAtom(_DisorderedAtom):
    """Disoreded Atom class derived from Biopython Disordered Atom and made compatible with
    OpenMM Atom."""

    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent.get_top_parent()

    def __getstate__(self):
        """Return state of the atom object for pickling, excluding neighbors 
        to avoid infinite recursion errors"""
        state = {k: v for k, v in self.__dict__.items() if k != "neighbors"}
        return state

    def __setstate__(self, state):
        """Set state of the atom object for pickling"""
        self.__dict__.update(state)

    @property
    def topo_definition(self):
        """Topology related parameters for the atom. This returns the selected 
        child's topology definition"""
        return self.selected_child.topo_definition

    @topo_definition.setter
    def topo_definition(self, atom_def):
        self.selected_child.topo_definition = atom_def
        for atom in self.child_dict.values():
            atom.topo_definition = atom_def
        