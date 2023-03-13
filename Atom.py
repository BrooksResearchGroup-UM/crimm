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
        super().__init__(
            name,
            coord,
            bfactor,
            occupancy,
            altloc,
            fullname,
            serial_number,
            element=element,
            pqr_charge=pqr_charge,
            radius=radius,
        )
        # Forcefield Parameters and Topology Definitions
        self._topo_def = None

    @property
    def topo_definition(self):
        """Topology related parameters for the atom"""
        return self._topo_def
    
    @topo_definition.setter
    def ff_params(self, atom_def):
        if not isinstance(atom_def):
            raise TypeError(
                'AtomDefinition class is need to set up forcefield paramerers'
            )
        if atom_def.name != self.name:
            warnings.warn(
                f"Atom Name Mismatch: Definition={atom_def.name} Atom={self.name}"
            )
        self._topo_def = atom_def
        self.id = self._topo_def.name

class DisorderedAtom(_DisorderedAtom):
    """Disoreded Atom class derived from Biopython Disordered Atom and made compatible with
    OpenMM Atom."""
    def __init__(self, id):
        super().__init__(id)
