"""Model class, used in Structure objects."""
import warnings
import crimm
from Bio.PDB.Model import Model as _Model
from crimm.StructEntities.Chain import PolymerChain
from crimm.Data.components_dict import NUCLEOSIDE_PHOS # Nucleoside phosphates and phosphonates

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
                        break
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
    
class OrganizedModel:
    """The OrganizedModel class represents a model in a structure with a specific 
    organization of chains. It is derived from the Model class and is used in the 
    Structure class to organize chains into categories. Require network access to
    fetch binding affinity and drugbank information."""
    def __init__(self, model: Model):
        self._model = model
        self.protein = []
        self.RNA = []
        self.DNA = []
        self.other_polymer = []
        self.phos_ligand =[]
        self.ligand = []
        self.co_solvent = []
        self.ion = []
        self.solvent = []
        self.unknown_type = []
        self.lig_names = set()
        self.binding_info = None
        if self._model.pdb_id is None:
            warnings.warn(
                "PDB ID not set for this model! Use model.set_pdb_id() to set it."
                "Binding affinity information will not be fetched. Possible errors "
                "in ligand classification."
            )
        else:
            self.binding_info = crimm.Fetchers.query_binding_affinity_info(self._model.pdb_id)
        if self.binding_info is not None:
            self.lig_names = set(self.binding_info.comp_id)
        self.organize()

    def is_ligand(self, resname):
        """Check if the residue is a ligand."""
        if len(self.lig_names) > 0:
            return resname in self.lig_names
        else:
            return crimm.Fetchers.query_drugbank_info(resname) is not None

    def organize(self):
        """Organize the chains in the model into categories."""
        _temp_ligand = []
        for chain in self._model.chains:
            if chain.chain_type == 'Polypeptide(L)':
                self.protein.append(chain)
            elif chain.chain_type == 'Polyribonucleotide':
                self.RNA.append(chain)
            elif chain.chain_type == 'Polydeoxyribonucleotide':
                self.DNA.append(chain)
            elif isinstance(chain, PolymerChain):
                self.other_polymer.append(chain)
            elif chain.chain_type == 'Solvent':
                self.solvent.append(chain)
            elif chain.chain_type == 'Heterogens':
                _temp_ligand.extend(chain.residues)
            else:
                self.unknown_type.append(chain)

        for res in _temp_ligand:
            if len(res.resname) == 2:
                self.ion.append(res)
            elif res.resname in NUCLEOSIDE_PHOS:
                self.phos_ligand.append(res)
            elif res.resname in self.lig_names:
                self.ligand.append(res)
            elif crimm.Fetchers.query_drugbank_info(res.resname) is not None:
                self.ligand.append(res)
            else:
                self.co_solvent.append(res)

    def __repr__(self):
        repr_str = f"<OrganizedModel model={self._model.get_id()} "
        if self._model.pdb_id is not None:
            repr_str += f"PDB ID={self._model.pdb_id} "
        if len(self.protein) > 0:
            repr_str += f"Protein={len(self.protein)} "
        if len(self.RNA) > 0:
            repr_str += f"RNA={len(self.RNA)} "
        if len(self.DNA) > 0:
            repr_str += f"DNA={len(self.DNA)} "
        if len(self.other_polymer) > 0:
            repr_str += f"OtherPolymer={len(self.other_polymer)} "
        if len(self.phos_ligand) > 0:
            repr_str += f"NucleosidePhosphates/Phosphonate={len(self.phos_ligand)} "
        if len(self.ligand) > 0:
            repr_str += f"Ligand={len(self.ligand)} "
        if len(self.co_solvent) > 0:
            repr_str += f"CoSolvent={len(self.co_solvent)} "
        if len(self.ion) > 0:
            repr_str += f"Ion={len(self.ion)} "
        if len(self.solvent) > 0:
            repr_str += f"Solvent={len(self.solvent)} "
        if len(self.unknown_type) > 0:
            repr_str += f"UnknownType={len(self.unknown_type)} "
        repr_str += ">"
        return repr_str
    
    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self._model) == 0:
            return
        from crimm.Visualization import show_nglview_multiple
        from IPython.display import display
        display(show_nglview_multiple(self._model.child_list))
        print(repr(self))
        print(self._model.expanded_view())
        