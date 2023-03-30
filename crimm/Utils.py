import os
import io
import requests
from crimm.IO import MMCIFParser

def find_local_cif_path(pdb_id, entry_point):
    """Find the path to a local cif file"""
    pdb_id = pdb_id.lower()
    subdir = pdb_id[1:3]
    file_path = os.path.join(entry_point, subdir, pdb_id+'.cif')
    if os.path.exists(file_path):
        return file_path

def get_pdb_entry(pdb_id, entry_point = "https://files.rcsb.org/download"):
    """Get a pdb entry from the rcsb database"""
    f_handle = io.StringIO()
    result = requests.get(
        f"{entry_point}/{pdb_id}.cif", 
        timeout=10
    )
    if (code := result.status_code) != 200:
        raise ValueError(
            "Request to rcsb did not return valid result! "
            f"[Status Code] {code}"
        )
    f_handle.write(result.content.decode('utf-8'))
    f_handle.seek(0)
    return f_handle

def get_structure(pdb_id, local_entry = None):
    """Get a structure from a pdb id or a local file"""
    if local_entry is not None:
        file = find_local_cif_path(pdb_id, entry_point = local_entry)
    else:
        file = get_pdb_entry(pdb_id)
    Parser = MMCIFParser(
        include_solvent=False
    )
    structure = Parser.get_structure(file)
    return structure