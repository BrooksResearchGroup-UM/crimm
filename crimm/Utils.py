import os
import io
import requests
import warnings
from crimm.IO import MMCIFParser
from crimm.Superimpose.ChainSuperimposer import ChainSuperimposer

def uniprot_id_query(pdbid, entity_id):
    """Query the RCSB PDB API for uniprot id for a given entity id and chain entity id"""
    rcsb_url = f"https://data.rcsb.org/rest/v1/core/uniprot/{pdbid}/{entity_id}"
    response = requests.get(rcsb_url, timeout=200)
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

def find_local_cif_path(pdb_id, entry_point):
    """Find the path to a local cif file"""
    pdb_id = pdb_id.lower()
    subdir = pdb_id[1:3]
    file_path = os.path.join(entry_point, subdir, pdb_id+'.cif')
    if os.path.exists(file_path):
        return file_path

def cif_from_url(cif_url):
    """Get a cif file from a url, return a file handle to the cif file"""
    f_handle = io.StringIO()
    result = requests.get(cif_url,timeout=10)
    if (code := result.status_code) != 200:
        warnings.warn(
            "GET request for mmcif file did not return valid result!\n"
            f"[Status Code] {code}"
        )
        return
    f_handle.write(result.content.decode('utf-8'))
    f_handle.seek(0)
    return f_handle

def get_alphafold_entry(uniprot_id):
    """Get a pdb entry from the alphafold database for a given uniprot id
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
    return cif_from_url(alphafold_cif_url)

def fetch_alphafold(pdb_id, entity_id):
    """Get a structure from the alphafold database for a given PDB id and entity id"""
    uniprot_id = uniprot_id_query(pdb_id, entity_id)
    msg = f"AlphaFold DB failed find structure for {pdb_id}-{entity_id}"
    if uniprot_id is None:
        # We only issue warning here
        warnings.warn(msg)
        return
    file = get_alphafold_entry(uniprot_id)
    if file is None:
        # We only issue warning here
        warnings.warn(msg)
        return
    parser = MMCIFParser(
        # AlphaFold only has one model and does not have multiple 
        # assemblies, solvent, or hydrogens
        first_model_only = True,
        first_assembly_only = False, 
        include_hydrogens = False,
        include_solvent = False
    )
    structure = parser.get_structure(file)
    return structure

def get_pdb_entry(pdb_id):
    """Get a pdb entry from the rcsb database"""
    entry_point = "https://files.rcsb.org/download"
    rcsb_cif_url = f"{entry_point}/{pdb_id}.cif"
    return cif_from_url(rcsb_cif_url)

def fetch(
        pdb_id,
        local_entry = None,
        first_model_only = True,
        first_assembly_only = True,
        include_solvent = True,
        include_hydrogens = False,
    ):
    """Get a structure from a pdb id or a local file"""
    if local_entry is not None:
        file = find_local_cif_path(pdb_id, entry_point = local_entry)
    else:
        file = get_pdb_entry(pdb_id)
    if file is None:
        raise ValueError(f"Could not load file for {pdb_id}")
    parser = MMCIFParser(
        first_model_only = first_model_only,
        first_assembly_only = first_assembly_only,
        include_hydrogens = include_hydrogens,
        include_solvent=include_solvent
    )
    structure = parser.get_structure(file)
    return structure

def fetch_alphafold_from_chain(chain):
    """Find the alphafold structure for a given chain. The chain must have a parent 
    structure with a valid PDB ID. The chain must also have an entity id assigned 
    from mmCIF.
    The returned AlphaFold structure will have only one model and one chain, and the
    chain will be superimposed to the input chain.
    """
    if not hasattr(chain.parent, 'parent') or chain.parent.parent is None:
        raise ValueError("Chain has no parent structure!")
    pdb_id = chain.parent.parent.id
    if pdb_id is None or len(pdb_id) != 4:
        raise ValueError(f"Chain's PDB ID {pdb_id} is invalid!")
    entity_id = chain.entity_id
    af_struct =  fetch_alphafold(pdb_id, entity_id)
    if af_struct is None:
        warnings.warn(
            f"Could not find AlphaFold structure for {pdb_id}-{entity_id}"
        )
        return
    af_chain = af_struct.models[0].chains[0]
    imposer = ChainSuperimposer()
    imposer.set_chains(chain, af_chain)
    imposer.apply_transform()
    return af_struct