"""Model class, used in Structure objects."""
import warnings
from Bio.PDB.Model import Model as _Model

class Model(_Model):
    """The extended Model class representing a model in a structure.
    Derived from Biopython's Bio.PDB.Model and compatible with Biopython functions

    In a structure derived from an X-ray crystallography experiment,
    only a single model will be present (with some exceptions). NMR
    structures normally contain many different models.
    """
    def __init__(self, id, serial_num=None):
        super().__init__(id, serial_num=serial_num)
        self.pdb_id = None
        self.connect_dict = {}
        self.connect_atoms = {}

    def set_pdb_id(self, pdb_id):
        """Set the PDB ID of this model."""
        if self.pdb_id is not None:
            warnings.warn(
                f"Overwriting PDB ID {self.pdb_id} with {pdb_id}"
            )
        self.pdb_id = pdb_id

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
        from IPython.display import display
        display(show_nglview_multiple(self.child_list))
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

    def set_connect(self, connect_dict):
        """Set the connect_dict attribute of this model. connect_dict is a dictionary of
        pairs of atom identified chain_id, resname, resseq, atom_id and altloc for 
        disulfide bonds, covalent bonds, metal coordination, saltbridge, etc."""
        self.connect_dict = connect_dict
        for connect_type, records in self.connect_dict.items():
            self.connect_atoms[connect_type] = []
            for record in records:
                atom_pairs = self._process_connect_record(record)
                if len(atom_pairs) != 2:
                    warnings.warn(
                        f"Insufficient atoms for connect record {connect_type}"
                    )
                    continue
                self.connect_atoms[connect_type].append(tuple(atom_pairs))

    def _process_connect_record(self, record):
        atom_pairs = []
        for atom_info in record:
            chain_id = atom_info['chain']
            resseq = atom_info['resseq']
            resname = str(atom_info['resname'])
            altloc = atom_info['altloc']
            atom_id = atom_info['atom_id']
            if chain_id not in self:
                warnings.warn(
                    f"Chain {chain_id} not found in model {self.get_id()}"
                )
                continue
            chain = self[chain_id]
            if chain.chain_type == 'Heterogens':
                for het_res in chain:
                    if het_res.resname == resname:
                        residue = het_res
            elif chain.chain_type == 'Solvent':
                # We simply cannot find which atom the record is referring to
                warnings.warn(
                    f"Chain {chain_id} is a solvent chain, connect atom "
                    "assignment skipped!"
                )
                continue
            elif resseq not in chain:
                warnings.warn(
                    f"Residue {resseq} not found in chain {chain_id}"
                )
                continue
            else:
                residue = chain[resseq]
            if residue.resname != resname:
                warnings.warn(
                    f"Residue {resseq} in chain {chain_id} has name "
                    f"{residue.resname}, not {resname}"
                )
                continue
            atom = residue[atom_id]
            if altloc:
                atom = atom.child_dict[altloc]
            atom_pairs.append(atom)
        return tuple(atom_pairs)
        