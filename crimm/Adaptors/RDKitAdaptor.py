import warnings
import json
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from crimm.IO import get_pdb_str

class LigandBondOrderException(Exception):
    """Define class LigandBondOrderException."""

class SmilesQueryException(Exception):
    """Define class SmilesQueryException."""

class LigandBondOrderWarning(Warning):
    """Define class LigandBondOrderException."""

class SmilesQueryWarning(Warning):
    """Define class SmilesQueryException."""

def _build_smiles_query(resname):
    query = '''
    {
        chem_comps(comp_ids: ["{var_lig_id}"]) {
            rcsb_id
            rcsb_chem_comp_descriptor {
            SMILES
            }
        }
    }
    '''.replace("{var_lig_id}", resname)
    return query

def query_rcsb_for_smiles(het_res):
    """Query the canonical SMILES for the heterogen molecule from PDB based on chem_comps
    ID"""
    url="https://data.rcsb.org/graphql"
    response = requests.post(
            url, json={'query': _build_smiles_query(het_res.resname)}, timeout=1000
        )
    response_dict = json.loads(response.text)
    return_vals = response_dict['data']['chem_comps']
    if not return_vals:
        warnings.warn(
            f'Query on {het_res.resname} did not return requested information: '
            'chem_comps (smiles string)'
        )
        return

    smiles = return_vals[0]['rcsb_chem_comp_descriptor']['SMILES']
    return smiles

def heterogen_to_rdkit(het_res, smiles=None):
    """Convert a heterogen to an rdkit mol. The bond orders are set based on the
    SMILES string. If the SMILES string is not provided, the function will query
    the RCSB PDB for the SMILES string.
    
    Args:
        het_res (Heterogen): The heterogen to be converted.
        smiles (str, optional): The SMILES string of the heterogen. Defaults to None.
        
    Returns:
        rdkit.Chem.rdchem.Mol: The rdkit mol of the heterogen.
    """
    mol = Chem.MolFromPDBBlock(get_pdb_str(het_res))
    if smiles is None:
        smiles = query_rcsb_for_smiles(het_res)
    # We do not allow rdkit mol return if the correct bond orders are not set
    if smiles is None:
        msg = (
            'Fail to set bond orders on the Ligand mol! PDB query on SMILES does not'
            'return any result.'
        )
        warnings.warn(msg, SmilesQueryWarning)
        return

    template = AllChem.MolFromSmiles(smiles)
    template = Chem.RemoveHs(template)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
        if het_res.pdbx_description is not None:
            mol.SetProp('Description', str(het_res.pdbx_description))
        return mol

    except ValueError:
        msg = (
            'No structure match found! Possibly the SMILES string supplied or reported on PDB '
            'mismatches the ligand structure.'
        )
        warnings.warn(msg, LigandBondOrderWarning)
        return