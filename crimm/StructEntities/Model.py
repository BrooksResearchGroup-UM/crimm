"""Model class, used in Structure objects."""

from Bio.PDB.Model import Model as _Model

class Model(_Model):
    """The extended Model class representing a model in a structure.
    Derived from Biopython's Bio.PDB.Model and compatible with Biopython functions

    In a structure derived from an X-ray crystallography experiment,
    only a single model will be present (with some exceptions). NMR
    structures normally contain many different models.
    """
    def __repr__(self):
        return f"<Model id={self.get_id()} Chains={len(self)}>" 
    
    def expanded_view(self):
        """Print the hierarchy tree of this model."""
        hierarchy_str = repr(self)
        branch_symbols = '\n\t│\n\t├───'
        for chain in self:
            hierarchy_str += branch_symbols
            hierarchy_str += "\n\t├──────".join(chain.expanded_view().split('\n  '))
        return hierarchy_str

    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from crimm.Visualization import show_nglview_multiple
        show_nglview_multiple(self.child_list)
        print(self.expanded_view())

    @property
    def chains(self):
        """Alias for child_list. Returns the list of chains in this model."""
        return self.child_list
    
    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent
    
    def reset_atom_serial_numbers(self, include_alt=True):
        """Reset all atom serial numbers in the encompassing entity (the parent 
        structure, if it exists) starting from 1."""
        i = 1
        for atom in self.get_atoms(include_alt=include_alt):
            atom.set_serial_number(i)
            i+=1
    
    def get_atoms(self, include_alt=False):
        """Return a generator of all atoms from this model. If include_alt is True, the 
        disordered residues will be expanded and altloc of disordered atoms will be included."""
        for chain in self:
            yield from chain.get_atoms(include_alt=include_alt)
