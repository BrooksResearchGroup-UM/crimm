import warnings
from Bio.PDB.Residue import Residue as _Residue
from Bio.PDB.Entity import Entity
from Bio.PDB.Residue import DisorderedResidue as _DisorderedResidue
from crimm.StructEntities.TopoElements import Bond

class Residue(_Residue):
    """Residue class derived from Biopython Residue and made compatible with
    CHARMM Topology."""
    def __init__(self, res_id, resname, segid, author_seq_id=None):
        super().__init__(res_id, resname, segid)
        self.author_seq_id = author_seq_id
        # Forcefield Parameters
        self.topo_definition = None
        self.missing_atoms = None
        self.missing_hydrogens = None
        self.atom_groups = None
        self.total_charge = None
        self.impropers = None
        self.cmap = None
        self.H_donors = None
        self.H_acceptors = None
        self.is_patch = None
        self.param_desc = None
        self.undefined_atoms = None
    
    @property
    def atoms(self):
        """Alias for child_list. Return the list of atoms in the residue."""
        return self.child_list
    
    def get_atoms(self, include_alt=False):
        """Return the list of all atoms. If include_alt is True, all altloc of 
        disordered atoms will be present."""
        if include_alt:
            yield from self.get_unpacked_list()
        else:
            yield from self.child_list
    
    def reset_atom_serial_numbers(self, include_alt=True):
        """Reset all atom serial numbers in the encompassing entity (the parent
        structure, model, and chain, if they exist) starting from 1."""
        top_parent = self.get_top_parent()
        if top_parent is not self:
            top_parent.reset_atom_serial_numbers(include_alt=include_alt)
            return
        # no parent, reset the serial number for the entity itself
        i = 1
        for atom in self.get_atoms(include_alt=include_alt):
            atom.set_serial_number(i)
            i+=1

    def _ipython_display_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from crimm.Visualization import show_nglview
        from IPython.display import display
        display(show_nglview(self))
        print(repr(self))

    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent.get_top_parent()
    
    def get_bonds_within_residue(self):
        """Return a list of bonds within the residue (peptide bonds linking neighbor 
        residues are excluded). Raise ValueError if the topology definition is 
        not loaded."""
        if self.topo_definition is None:
            raise ValueError(
                'Topology definition is not loaded for this residue!'
            )
        bonds = []
        bond_dict = self.topo_definition.bonds
        for bond_type, bond_list in bond_dict.items():
            for atom_name1, atom_name2 in bond_list:
                if not (atom_name1 in self and atom_name2 in self):
                    continue
                atom1, atom2 = self[atom_name1], self[atom_name2]
                bonds.append(
                    Bond(atom1, atom2, bond_type)
                )
        return bonds

class DisorderedResidue(_DisorderedResidue):
    
    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent.get_top_parent()
    
    def reset_atom_serial_numbers(self, include_alt=True):
        """Reset all atom serial numbers in the encompassing entity (the parent
        structure, model, and chain, if they exist) starting from 1."""
        self.selected_child.reset_atom_serial_numbers(include_alt=include_alt)
        
    def get_atoms(self, include_alt=False):
        """Return a generator of all atoms in the disordered residue."""
        if include_alt:
            for res in self.child_dict.values():
                yield from res.get_unpacked_list()
        else:
            yield from self.selected_child.child_list


class Heterogen(Residue):
    def __init__(self, res_id, resname, segid):
        super().__init__(res_id, resname, segid)
        self.pdbx_description = None

    def add(self, atom):
        """Special method for Add an Atom object to Heterogen. Any duplicated 
        Atom id will be renamed.

        Checks for adding duplicate atoms, and raises a warning if so.
        """
        atom_id = atom.get_id()
        if self.has_id(atom_id):
            # some ligands in PDB could have duplicated atom names, we will
            # recursively check and rename the atom.
            atom.id = atom.id+'A'
            warnings.warn(
                f"Atom {atom_id} defined twice in residue {self}!"+
                f' Atom id renamed to {atom.id}.'
            )
            self.add(atom)
        else:
            Entity.add(self, atom)