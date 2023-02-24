import warnings
from Bio.PDB.Model import Model as _Model
from NGLVisualization import load_nglview_multiple


class Model(_Model):
    def __init__(self, id, serial_num=None):
        super().__init__(id, serial_num)
    
    def __repr__(self):
        hierarchy_str = f"<Model id={self.get_id()} Chains={len(self)}>\n\t|"
        for chain in self:
            hierarchy_str+='\n\t|---'+chain.__repr__()
        return hierarchy_str
    
    def _repr_html_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from IPython.display import display
        self.reset_atom_serial_numbers
        view = load_nglview_multiple(self)
        display(view)

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

    def get_pdb_str(self, reset_serial = True, include_alt = True):
        if reset_serial:
            self.reset_atom_serial_numbers()
        pdb_str = ''
        for chain in self:
            pdb_str += chain.get_pdb_str(
                    reset_serial = False, 
                    include_alt = include_alt
                )
        return pdb_str
