from Bio.PDB.Model import Model as _Model
import warnings

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

        try:
            import nglview as nv
        except ImportError:
            warnings.warn(
                "WARNING: nglview not found! Install nglview to show\
                protein structures. \
                http://nglviewer.org/nglview/latest/index.html#installation"
            )
            return self.__repr__()
        view = nv.NGLWidget()
        for chain in self:
            chain.load_nglview(view)
        display(view)