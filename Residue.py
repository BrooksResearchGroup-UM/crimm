import warnings
import json
import requests
from Bio.PDB.Residue import Residue as _Residue
from Bio.PDB.Entity import Entity
from Bio.PDB.Residue import DisorderedResidue as _DisorderedResidue
from Bio.PDB import PDBIO
from ResidueExceptions import LigandBondOrderWarning, SmilesQueryWarning
from TopoDefinitions import ResidueDefinition
from TopoElements import Bond
from NGLVisualization import load_nglview

class Residue(_Residue):
    """Residue class derived from Biopython Residue and made compatible with
    CHARMM Topology."""
    def __init__(self, res_id, resname, segid):
        super().__init__(res_id, resname, segid)
        # Forcefield Parameters
        self.topo_definition: ResidueDefinition = None
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
    
    def get_unpacked_atoms(self):
        """Return the list of all atoms where the all altloc of disordered atoms will
        be present."""
        return self.get_unpacked_list()
    
    def reset_atom_serial_numbers(self, include_alt=True):
        """Reset all atom serial numbers in the encompassing entity (the parent
        structure, model, and chain, if they exist) starting from 1."""
        top_parent = self.get_top_parent()
        if top_parent is not self:
            top_parent.reset_atom_serial_numbers(include_alt=include_alt)
            return
        # no parent, reset the serial number for the entity itself
        i = 1
        if include_alt:
            all_atoms = self.get_unpacked_atoms()
        else:
            all_atoms = []
            for atom in self:
                if atom.disorderd:
                    atom = atom.selected_child
                all_atoms.append(atom)
        for atom in all_atoms:
            atom.serial_number = i
            i+=1

    def _repr_html_(self):
        """Return the nglview interactive visualization window"""
        if len(self) == 0:
            return
        from IPython.display import display
        view = load_nglview(self)
        display(view)
        return
    
    @staticmethod
    def _get_child(parent, include_alt):
        if include_alt:
            return parent.get_unpacked_list()
        return parent.child_list

    def get_top_parent(self):
        if self.parent is None:
            return self
        return self.parent.get_top_parent()
    
    def get_pdb_str(self, include_alt = True):
        if self.parent is not None:
            chain = self.get_parent()
            chain_id = chain.get_id()
        else:
            chain_id = '_'

        io = PDBIO()
        pdb_string = ''
        hetfield, resseq, icode = self.id
        resname = self.resname
        segid = self.segid

        atom_serial = 1
        for atom in self._get_child(self, include_alt):
            atom_line = io._get_atom_line(
                atom,
                hetfield,
                segid,
                atom_serial,
                resname,
                resseq,
                icode,
                chain_id,
            )
            pdb_string += atom_line
            atom_serial += 1
        return pdb_string

    ##TODO: move this to a separate class for structure modeling and preparations
    def load_topo_definition(self, res_def: ResidueDefinition, QUIET=False):
        """Load topology definition to the residue and find any missing atoms."""
        if not isinstance(res_def, ResidueDefinition):
            raise TypeError(
                'ResidueDefinition class is required to set up topology info!'
            )
        self.topo_definition = res_def
        self._load_atom_groups()
        self.total_charge = res_def.total_charge
        self.impropers = self.topo_definition.impropers
        self.cmap = self.topo_definition.cmap
        self.H_donors = self.topo_definition.H_donors
        self.H_acceptors = self.topo_definition.H_acceptors
        self.param_desc = res_def.desc
        self.undefined_atoms = []
        for atom in self:
            if atom.name not in self.topo_definition:
                if not QUIET:
                    warnings.warn(
                        f"Atom {atom.name} is not defined in the topology file!"
                    )
                self.undefined_atoms.append(atom)

    def _bifurcate_missing_atom(self, atom: str):
        """Separate missing heavy atoms and missing hydrogen atom by atom name"""
        if atom.startswith('H'):
            self.missing_hydrogens.append(atom)
        else:
            self.missing_atoms.append(atom)
    
    def _load_group_atom_topo_definition(self, atom_name_list) -> list:
        """Load topology definition to each atom in the residue and find any missing 
        atoms. 
        Argument:
            atom_name_list: list of atom names that are in the same group
        Return:
            atom_group: the Atom object in the group
        """
        atom_group = []
        for atom_name in atom_name_list:
            if atom_name not in self:
                self._bifurcate_missing_atom(atom_name)
                continue
            cur_atom = self[atom_name]
            cur_atom.topo_definition = self.topo_definition[atom_name]
            atom_group.append(cur_atom)
        return atom_group

    def _load_atom_groups(self):
        self.atom_groups = {} 
        self.missing_atoms, self.missing_hydrogens = [],[]
        atom_groups_dict = self.topo_definition.atom_groups
        for group_num, atom_names in atom_groups_dict.items():
            cur_group = self._load_group_atom_topo_definition(atom_names)
            self.atom_groups.update({group_num:cur_group})
    
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
    
    def reset_atom_serial_numbers(self):
        i = 1
        for res in self.child_dict.values():
            for atom in res.get_unpacked_list():
                atom.serial_number = i
                i += 1
        
    def get_unpacked_atoms(self):
        atoms = []
        for res in self.child_dict.values():
            atoms.extend(res.get_unpacked_list())
        return atoms

    def get_pdb_str(self, reset_serial = True, include_alt = True):
        # if no alternative coords/residues are not included
        # we only return the pdb str for the selected child
        if not include_alt:
            return self.selected_child.get_pdb_str(
                    reset_serial = reset_serial, 
                    include_alt = False
                )
        
        # else we return all alternative res/atoms
        pdb_str = ''
        if reset_serial:
            self.reset_atom_serial_numbers()
        for res in self.child_dict.values():
            pdb_str += res.get_pdb_str(
                    reset_serial=False, 
                    include_alt=True
                )
        
        return pdb_str

    def load_topo_definition(self, res_def: ResidueDefinition, QUIET=False):
        """Load topology definition to the residue and find any missing atoms."""
        for res in self.child_dict.values():
            res.load_topo_definition(res_def, QUIET=QUIET)

class Heterogen(Residue):
    def __init__(self, res_id, resname, segid):
        super().__init__(res_id, resname, segid)
        self.smiles = None
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

    def _build_smiles_query(self):
        query = '''
        {
            chem_comps(comp_ids: ["{var_lig_id}"]) {
                rcsb_id
                rcsb_chem_comp_descriptor {
                SMILES
                }
            }
        }
        '''.replace("{var_lig_id}", self.resname)
        return query

    def query_rcsb_for_smiles(self):
        """Query the canonical SMILES for the molecule from PDB based on chem_comps
        ID"""
        url="https://data.rcsb.org/graphql"
        response = requests.post(
                url, json={'query': self._build_smiles_query()}, timeout=1000
            )
        response_dict = json.loads(response.text)
        return_vals = response_dict['data']['chem_comps']
        if not return_vals:
            warnings.warn(
                f'Query on {self.resname} did not return requested information: '
                'chem_comps (smiles string)'
            )
            return

        smiles = return_vals[0]['rcsb_chem_comp_descriptor']['SMILES']
        return smiles
            
    
    def to_rdkit(self, smiles = None):
        """Convert the molecule to an rdkit mol object. SMILES string is required to
        set the correct bond orders on the molecule. If no SMILES string is supplied
        as a parameter, rcsb PDB will be queried based on the molecule's chem_comp ID 
        for its canonical SMILES. 
        Note: The query might not return the correct SMILES. In this case, a exception
        will be raised."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError('Rdkit is required for conversion to rdkit Mol')

        mol = Chem.MolFromPDBBlock(self.get_pdb_str())
        if smiles is None:
            self.smiles = self.query_rcsb_for_smiles()
        else:
            self.smiles = smiles
        # We do not allow rdkit mol return if the correct bond orders are not set
        if self.smiles is None:
            msg = (
                'Fail to set bond orders on the Ligand mol! PDB query on SMILES does not'
                'return any result.'
            )
            warnings.warn(msg, SmilesQueryWarning)
            return

        template = AllChem.MolFromSmiles(self.smiles)
        template = Chem.RemoveHs(template)
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
            if self.pdbx_description is not None:
                mol.SetProp('Description', str(self.pdbx_description))
            return mol

        except ValueError:
            msg = (
                'No structure match found! Possibly the SMILES string supplied or reported on PDB '
                'mismatches the ligand structure.'
            )
            warnings.warn(msg, LigandBondOrderWarning)
            return
