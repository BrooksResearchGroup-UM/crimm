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
        
        # Forcefield Parameters
        self.atom_group = None
        self.atom_type = None
        self.is_donor = None
        self.is_acceptor = None
        self.charge = None

    @staticmethod
    def _find_parent_model(entity):
        """Find the Model/Topology this Atom belongs to"""
        parent = entity.parent
        if not hasattr(parent, 'level'):
            return None
        if parent.level == 'M':
            return parent
        return Atom._find_parent_model(parent)

class DisorderedAtom(_DisorderedAtom):
    """Disoreded Atom class derived from Biopython Disordered Atom and made compatible with
    OpenMM Atom."""
    def __init__(self, id):
        super().__init__(id)
