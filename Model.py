import warnings
from Bio.PDB.Model import Model as _Model
from NGLVisualization import load_nglview_multiple

class Model(_Model):
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

    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent
    
    def reset_atom_serial_numbers(self):
        i = 1
        for atom in self.get_unpacked_atoms():
            atom.serial_number = i
            i+=1

    def get_unpacked_atoms(self):
        atoms = []
        for chain in self:
            atoms.extend(chain.get_unpacked_atoms())
        return atoms

    def get_pdb_str(self, include_alt = True, reset_serial = True):
        if reset_serial:
            self.reset_atom_serial_numbers()
        pdb_str = ''
        for chain in self:
            pdb_str += chain.get_pdb_str(
                include_alt = include_alt, reset_serial = False
            )
        return pdb_str
