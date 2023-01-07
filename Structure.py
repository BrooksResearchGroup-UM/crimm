from Bio.PDB.Structure import Structure as _Structure

class Structure(_Structure):
    def __init__(self, id) -> None:
        super().__init__(id)

    def __repr__(self):
        hierarchy_str = f"<Structure id={self.get_id()} Models={len(self)}>"
        if len(self) == 0:
            return hierarchy_str
        first_model = self.child_list[0]
        hierarchy_str+='\n|\n|---'+first_model.__repr__()
        if len(self) > 1:
            hierarchy_str+=f'\n[{len(self)-1} models truncated ...]'
        return hierarchy_str
    
    def _repr_html_(self):
        if len(self) == 0:
            return
        return self.child_list[0]._repr_html_()