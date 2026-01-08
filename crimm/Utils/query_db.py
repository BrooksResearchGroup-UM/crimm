import warnings
import requests
import pandas as pd

def uniprot_id_query(pdbid, entity_id):
    """Query wth the RCSB PDB API for uniprot id for a given pdb id and 
    chain entity id
    """
    rcsb_url = f"https://data.rcsb.org/rest/v1/core/uniprot/{pdbid}/{entity_id}"
    response = requests.get(rcsb_url, timeout=500)
    if (code := response.status_code) != 200:
        warnings.warn(
            f"GET request on RCSB for \"{pdbid}-{entity_id}\" for uniprot ID "
            f"did not return valid result! \n[Status Code] {code}"
        )
        return
    kw1 = 'rcsb_uniprot_container_identifiers'
    kw2 = 'uniprot_id'
    uniprot_id = response.json()[0][kw1][kw2]
    return uniprot_id

def query_drugbank_info(lig_id):
    """Query the RCSB database for Drugbank info on a given ligand ID"""
    if len(lig_id) != 3:
        raise ValueError('Ligand ID has to be a three-letter code')

    query_url = f'https://data.rcsb.org/rest/v1/core/drugbank/{lig_id}'
    response = requests.get(query_url, timeout=500)
    if (code := response.status_code) != 200:
        warnings.warn(
            f"GET request on RCSB for \"{lig_id}\" for Drugbank Info "
            f"did not return valid result! \n[Status Code] {code}"
        )
        return
    return response.json()

def organize_drugbank_info(info_dict):
    """Organize the Drugbank info dictionary into a more readable format
    Args: info_dict (dict): The dictionary returned by the RCSB REST API
    Returns: drugbank_id, info, drug_products, targets"""
    if info_dict is None:
        warnings.warn(
            "No Drugbank Info provided!"
        )
        return None, None, None, None
    drugbank_id = info_dict['drugbank_container_identifiers'].get('drugbank_id')
    info = info_dict['drugbank_info']
    targets = info_dict.get('drugbank_target')
    drug_products = None
    if 'drug_products' in info:
        drug_products = info.pop('drug_products')
        drug_products = pd.DataFrame(drug_products)
    if targets:
        targets = pd.DataFrame(targets)
    return drugbank_id, info, drug_products, targets

def get_rcsb_web_data(pdb_id):
    """Get the data from the RCSB REST API for a given PDB ID
    Args: pdb_id (str): The PDB ID to query
    Returns: The dictionary from the RCSB REST API"""
    query_url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
    response = requests.get(query_url, timeout=500)
    if (code := response.status_code) != 200:
        warnings.warn(
            f"GET request on RCSB for \"{pdb_id}\" for binding affinity data "
            f"did not return valid result! \n[Status Code] {code}"
        )
        return
    return response.json()

def query_binding_affinity_info(pdb_id):
    """Query the RCSB PDB database for binding affinity info on a given PDB ID
    Args: pdb_id (str): The PDB ID to query
    Returns: The binding affinity info as a pandas DataFrame"""
    data = get_rcsb_web_data(pdb_id)
    keyword = 'rcsb_binding_affinity'
    if data is None:
        raise ValueError(
            f"Failed to fetch data for {pdb_id}. Check the PDB ID or network status.")
    if keyword not in data:
        return
    return pd.DataFrame(data[keyword])