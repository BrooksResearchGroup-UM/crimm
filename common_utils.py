import os
import io
import requests
from ChMMCIFParser import ChMMCIFParser as ChmParser

def find_local_cif_path(pdb_id, entry_point):
    pdb_id = pdb_id.lower()
    subdir = pdb_id[1:3]
    file_path = os.path.join(entry_point, subdir, pdb_id+'.cif')
    if os.path.exists(file_path):
        return file_path

def get_pdb_entry(pdb_id):
    f_handle = io.StringIO()
    result = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif")
    if (code := result.status_code) != 200:
        raise ValueError(
            "Request to rcsb did not return valid result! "
            f"[Status Code] {code}"
        )
    f_handle.write(result.content.decode('utf-8'))
    f_handle.seek(0)
    return f_handle

def get_structure(pdb_id, local_entry = None):
    if local_entry is not None:
        file = find_local_cif_path(pdb_id, entry_point = local_entry)
    else:
        file = get_pdb_entry(pdb_id)
    Parser = ChmParser(
        include_solvent=False
    )
    structure = Parser.get_structure(file)
    return structure