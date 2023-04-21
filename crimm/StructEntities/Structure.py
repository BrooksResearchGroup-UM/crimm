"""The structure class, representing a macromolecular structure."""

from Bio.PDB.Structure import Structure as _Structure

class Structure(_Structure):
    """The extended Structure class contains a collection of Model instances.
    Derived from Biopython's Bio.PDB.Model and compatible with Biopython functions
    """

    def __init__(self, id) -> None:
        super().__init__(id)
        self.header = None
        self.resolution = None
        self.method = None
        self.assemblies = None
        self.cell_info = None

    def __repr__(self):
        hierarchy_str = f"<Structure id={self.get_id()} Models={len(self)}>"
        if len(self) == 0:
            return hierarchy_str
        first_model = self.child_list[0]
        hierarchy_str+='\n│\n├───'+first_model.__repr__()
        if len(self) > 1:
            hierarchy_str+=f'\n[{len(self)-1} models truncated ...]'
        return hierarchy_str
    
    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        return self.child_list[0]._ipython_display_()

    @property
    def models(self):
        """Alias for child_list. Returns the list of models in this structure."""
        return self.child_list

    def get_atoms(self, include_alt=False):
        """Return a generator of all atoms from this structure. If include_alt is True, the 
        disordered residues will be expanded and altloc of disordered atoms will be included."""
        for model in self:
            yield from model.get_atoms(include_alt=include_alt)

    def reset_atom_serial_numbers(self, include_alt=True):
        """Reset all atom serial numbers in the structure starting from 1."""
        for model in self:
            model.reset_atom_serial_numbers(include_alt=include_alt)