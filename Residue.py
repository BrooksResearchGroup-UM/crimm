from Bio.PDB.Residue import Residue as _Residue
from Bio.PDB.Residue import DisorderedResidue as _DisorderedResidue
from Bio.PDB import PDBIO
from ResidueExceptions import LigandBondOrderException, SmilesQueryException
import warnings
import json
import requests

class Residue(_Residue):
    def __init__(self, id, resname, segid):
        super().__init__(id, resname, segid)

    def get_unpacked_atoms(self):
        return self.get_unpacked_list()
    
    def reset_atom_serial_numbers(self):
        i = 1
        for atom in self.get_unpacked_list():
            atom.serial_number = i
            i += 1

    def get_pdb_str(self, reset_serial = True, include_alt = True):
        if reset_serial:
            self.reset_atom_serial_numbers()
        
        if hasattr(self, "parent"):
            chain = self.get_parent()
            chain_id = chain.get_id()
        else:
            chain_id = 'XX'

        io = PDBIO()
        pdb_string = ''
        hetfield, resseq, icode = self.id
        resname = self.resname
        segid = self.segid
        if include_alt:
            get_child = lambda x: x.get_unpacked_list()
        else:
            get_child = lambda x: x.child_list

        for atom in get_child(self):
            atom_number = atom.serial_number
            s = io._get_atom_line(
                atom,
                hetfield,
                segid,
                atom_number,
                resname,
                resseq,
                icode,
                chain_id,
            )
            pdb_string += s
        return pdb_string
        
    
class DisorderedResidue(_DisorderedResidue):
    def __init__(self, id):
        super().__init__(id)

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

class Ligand(Residue):
    def __init__(self, id, resname, segid):
        super().__init__(id, resname, segid)

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
        '''.replace("{var_lig_id}",self.resname)
        return query

    def query_pdb_for_smiles(self):
        url="https://data.rcsb.org/graphql"
        r = requests.post(url, json={'query': self._build_smiles_query()})
        r_dict = json.loads(r.text)
        return_vals = r_dict['data']['chem_comps']
        if not return_vals:
            warnings.warn(
                'Query on {} did not return requested information: '
                'chem_comps (smiles string)'.format(lig_id)
            )
            return

        smiles = return_vals[0]['rcsb_chem_comp_descriptor']['SMILES']
        return smiles
            
    
    def to_rdkit(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError('Rdkit is required for conversion to rdkit Mol')

        mol = Chem.MolFromPDBBlock(self.get_pdb_str())
        smiles = self.query_pdb_for_smiles()
        # We do not allow rdkit mol return if the correct bond orders are not set
        if smiles is None:
            raise SmilesQueryException(
                'Fail to set bond orders on the Ligand mol! PDB query on SMILES does not'
                'return any result.'
            )

        template = AllChem.MolFromSmiles(smiles)
        template = Chem.RemoveHs(template)
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
        except ValueError:
            raise LigandBondOrderException(
                'No structure match found! Possibly the SMILES string reported on PDB ' 
                'mismatches the ligand structure.'
            )
        
        return mol