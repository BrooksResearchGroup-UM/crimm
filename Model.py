"""Model class, used in Structure objects."""

from Bio.PDB.Model import Model as _Model
from NGLVisualization import load_nglview_multiple

class Model(_Model):
    """The extended Model class representing a model in a structure.
    Derived from Biopython's Bio.PDB.Model and compatible with Biopython functions

    In a structure derived from an X-ray crystallography experiment,
    only a single model will be present (with some exceptions). NMR
    structures normally contain many different models.
    """
    def __repr__(self):
        hierarchy_str = f"<Model id={self.get_id()} Chains={len(self)}>"
        branch_symbols = '\n\t│\n\t├───'
        for chain in self:
            hierarchy_str += branch_symbols
            hierarchy_str += "\n\t├──────".join(chain.__repr__().split('\n  '))
        return hierarchy_str
    
    def _repr_html_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from IPython.display import display
        view = load_nglview_multiple(self)
        display(view)

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
        if include_alt:
            all_atoms = self.get_unpacked_atoms()
        else:
            all_atoms = self.get_atoms()
        for atom in all_atoms:
            atom.set_serial_number(i)
            i+=1

    def get_unpacked_atoms(self):
        """Return the list of all atoms from this model where the all altloc of 
        disordered atoms will be present."""
        atoms = []
        for chain in self:
            atoms.extend(chain.get_unpacked_atoms())
        return atoms
    
    def get_atoms(self):
        atoms = []
        for chain in self:
            atoms.extend(chain.get_atoms())
        return atoms

    def get_pdb_str(self, include_alt = True, reset_serial = True):
        if reset_serial:
            self.reset_atom_serial_numbers(include_alt=include_alt)
        pdb_str = ''
        for chain in self:
            pdb_str += chain.get_pdb_str(
                include_alt = include_alt, reset_serial = False
            )
        return pdb_str
