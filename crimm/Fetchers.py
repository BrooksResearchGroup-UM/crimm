import os
import io
import requests
import warnings
import pandas as pd
from Bio.Seq import Seq
from crimm.IO import MMCIFParser, PDBParser
from crimm.IO.MMCIF2Dict import MMCIF2Dict
from crimm.Superimpose.ChainSuperimposer import ChainSuperimposer

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

def _find_local_cif_path(pdb_id, entry_point):
    """Find the path to a local cif file"""
    pdb_id = pdb_id.lower()
    subdir = pdb_id[1:3]
    file_path = os.path.join(entry_point, subdir, pdb_id+'.cif')
    if os.path.exists(file_path):
        return file_path

def _file_handle_from_url(cif_url):
    """Get a cif file from a url, return a file handle to the cif file"""
    f_handle = io.StringIO()
    result = requests.get(cif_url,timeout=500)
    if (code := result.status_code) != 200:
        warnings.warn(
            "GET request for file did not return valid result!\n"
            f"[Status Code] {code}"
        )
        return
    f_handle.write(result.content.decode('utf-8'))
    f_handle.seek(0)
    return f_handle

def _get_alphafold_fh(uniprot_id):
    """Get a mmcif file handle from the alphafold database for a given uniprot id
    """
    query_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    response = requests.get(query_url, timeout=500)
    if (code := response.status_code) != 200:
        warnings.warn(
            f"GET request on AlphaFold DB for \"{uniprot_id}\"  did not return "
            f"valid result! \n[Status Code] {code}"
        )
        return
    alphafold_cif_url = response.json()[0]['cifUrl']
    return _file_handle_from_url(alphafold_cif_url)

def fetch_alphafold(uniprot_id):
    """Get a structure from the alphafold database for a given uniprot id"""
    msg = f"AlphaFold DB failed find structure for {uniprot_id}"
    if uniprot_id is None:
        # We only issue warning here
        warnings.warn(msg)
        return
    file = _get_alphafold_fh(uniprot_id)
    if file is None:
        # We only issue warning here
        warnings.warn(msg)
        return
    parser = MMCIFParser(
        # AlphaFold only has one model and does not have multiple 
        # assemblies, solvent, or hydrogens
        first_model_only = True,
        use_bio_assembly = False,
        include_hydrogens = False,
        include_solvent = False
    )
    structure = parser.get_structure(file)
    return structure

def _get_mmcif_fh(pdb_id):
    """Get a mmcif file handle from the rcsb database"""
    if len(pdb_id) == 4:
        entry_point = "https://files.rcsb.org/download"
    elif len(pdb_id) == 3:
        entry_point = "https://files.rcsb.org/ligands/download"
    else:
        raise ValueError(f"Invalid PDB ID {pdb_id}")
    rcsb_cif_url = f"{entry_point}/{pdb_id}.cif"
    return _file_handle_from_url(rcsb_cif_url)

def fetch_rcsb_as_dict(pdb_id):
    """Get info about a pdb entry as a dictionary from rcsb"""
    file = _get_mmcif_fh(pdb_id)
    if file is None:
        raise ValueError(f"Could not load file for {pdb_id}")
    cifdict = MMCIF2Dict(file)
    return cifdict

def fetch_rcsb(
        pdb_id,
        local_entry = None,
        first_model_only = True,
        use_bio_assembly = True,
        include_solvent = True,
        include_hydrogens = False,
    ):
    """Get a structure from rcsb with a pdb id or from a local mmcif file"""
    if len(pdb_id) == 3:
        raise ValueError("Ligand entries are not supported yet!")
    if local_entry is not None:
        file = _find_local_cif_path(pdb_id, entry_point = local_entry)
    else:
        file = _get_mmcif_fh(pdb_id)
    if file is None:
        raise ValueError(f"Could not load file for {pdb_id}")
    parser = MMCIFParser(
        first_model_only = first_model_only,
        use_bio_assembly = use_bio_assembly,
        include_hydrogens = include_hydrogens,
        include_solvent=include_solvent
    )
    structure = parser.get_structure(file)
    return structure

def fetch_swiss_model(uniprot_id):
    """Get the first matching stucuture from the Swiss Model database for a given 
    uniprot id
    """
    base_url = (
        "https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.{ext}"
        "?provider=swissmodel"
    )
    header_url = base_url.format(uniprot_id = uniprot_id, ext = 'json')
    struct_url = base_url.format(uniprot_id = uniprot_id, ext = 'pdb')
    result = requests.get(header_url,timeout=500)
    if (code := result.status_code) != 200:
        warnings.warn(
            "GET request for header did not return valid result!\n"
            f"[Status Code] {code}"
        )
        return
    info_dict = result.json()['result']
    can_seq_str = info_dict['sequence']
    chain_info_dict = info_dict['structures'][0]['chains'][0]
    desc = chain_info_dict['segments'][0]['smtl']['description']

    parser = PDBParser()
    struct_fh = _file_handle_from_url(struct_url)
    if struct_fh is None:
        return
    structure = parser.get_structure(struct_fh, f'{uniprot_id}-SwissModel')

    for chain in structure.child_list[0]:
        if not hasattr(chain, 'seq'):
            continue
        if str(chain.seq) in can_seq_str:
            chain.can_seq = Seq(can_seq_str)
            chain.pdbx_description = desc
    return structure

def fetch_swiss_model_multiple(uniprot_id):
    """Get all stucutures from the Swiss Model database for a given uniprot id
    """
    header_url = (
        f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.json"
        "?provider=swissmodel"
    )
    result = requests.get(header_url,timeout=500)
    if (code := result.status_code) != 200:
        warnings.warn(
            "GET request for header did not return valid result!\n"
            f"[Status Code] {code}"
        )
        return

    info_dict = result.json()['result']
    can_seq_str = info_dict['sequence']
    chain_info_dict = info_dict['structures'][0]['chains'][0]
    desc = chain_info_dict['segments'][0]['smtl']['description']

    struct_info_dicts = result.json()['result']['structures']
    structures = []
    for info_dict in struct_info_dicts:
        structures.append(
            _sm_struct_from_info_dict(uniprot_id, info_dict, can_seq_str, desc)
        )
    return structures

def _sm_struct_from_info_dict(uniprot_id, struct_info_dict, can_seq_str, desc):
    parser = PDBParser()
    structure_url = struct_info_dict['coordinates']
    struct_fh = _file_handle_from_url(structure_url)
    if struct_fh is None:
        return
    structure = parser.get_structure(struct_fh, f'{uniprot_id}-SwissModel')

    for chain in structure.models[0]:
        if not hasattr(chain, 'seq'):
            continue
        if str(chain.seq) in can_seq_str:
            chain.can_seq = Seq(can_seq_str)
            chain.pdbx_description = desc

    return structure

def _fetch_with_chain(chain, fetcher):
    if not hasattr(chain.parent, 'parent') or chain.parent.parent is None:
        raise ValueError("Chain has no parent structure!")
    pdb_id = chain.parent.parent.id
    if pdb_id is None or len(pdb_id) != 4:
        raise ValueError(f"Chain's PDB ID {pdb_id} is invalid!")
    entity_id = chain.entity_id
    uniprot_id = uniprot_id_query(pdb_id, entity_id)
    new_struct =  fetcher(uniprot_id)
    if new_struct is None:
        warnings.warn(
            f"Could not find AlphaFold structure for {pdb_id}-{entity_id}"
        )
        return
    new_chain = new_struct.models[0].chains[0]
    imposer = ChainSuperimposer()
    imposer.set_chains(chain, new_chain)
    imposer.apply_transform(new_struct)
    return new_struct

def fetch_alphafold_from_chain(chain):
    """Find the alphafold structure for a given chain. The chain must have a parent 
    structure with a valid PDB ID. The chain must also have an entity id assigned 
    from mmCIF.
    The returned AlphaFold structure will have only one model and one chain, and the
    chain will be superimposed to the input chain.
    """
    return _fetch_with_chain(chain, fetch_alphafold)

def fetch_swiss_model_from_chain(chain):
    """Find the Swiss Model structure for a given chain. The chain must have a parent 
    structure with a valid PDB ID. The chain must also have an entity id assigned 
    from mmCIF.
    The returned Swiss Model chain will be superimposed to the input chain.
    """
    return _fetch_with_chain(chain, fetch_swiss_model)

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