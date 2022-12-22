import os
from Bio.PDB.MMCIFParser import MMCIFParser
from PeptideChain import StandardChain as SChain
from Bio.PDB.Polypeptide import protein_letters_3to1_extended
from ChainExceptions import ChainConstructionException

def find_local_cif_path(pdb_id):
    pdb_id = pdb_id.lower()
    entry_point = '/mnt/backup/PDB/'
    subdir = pdb_id[1:3]
    file_path = os.path.join(entry_point, subdir, pdb_id+'.cif')
    if os.path.exists(file_path):
        return file_path

def get_first_chain_and_dict(pdb_id):
    parser = MMCIFParser(auth_chains=True, auth_residues=False, QUIET=True)
    structure = parser.get_structure(pdb_id, find_local_cif_path(pdb_id))
    cifdict = parser._mmcif_dict
    first_peptide_chain = list(create_entity_chain_dict(cifdict).keys())[0]
    chain = structure.child_list[0][first_peptide_chain]
    
    return chain, cifdict

def create_entity_chain_dict(cifdict):
    
    prefix = '_entity_poly'
    entity_ids = cifdict[prefix+'.entity_id']
    entity_types = cifdict[prefix+'.type']
    chain_ids = cifdict[prefix+'.pdbx_strand_id']
    known_seqs = cifdict[prefix+'.pdbx_seq_one_letter_code']
    can_seqs = cifdict[prefix+'.pdbx_seq_one_letter_code_can']
    chain_formatter = lambda x: ''.join(x.split('\n'))
    
    all_data = (entity_ids, entity_types, chain_ids, known_seqs, can_seqs)
    entity_chain_dict = dict()
    for entity_id, entity_type, chain_id_str, known_seq, can_seq in zip(*all_data):
        if entity_type != 'polypeptide(L)':
            continue
        chain_id_list = chain_id_str.split(',')
        
        for chain_id in chain_id_list:
            entity_chain_dict[chain_id] = (int(entity_id), chain_formatter(known_seq), chain_formatter(can_seq))
    
    return entity_chain_dict

def create_all_res_dict(cifdict):
    all_entity_id = (int(x) for x in cifdict["_entity_poly_seq.entity_id"])
    all_seq_id = (int(x) for x in cifdict["_entity_poly_seq.num"])
    all_res_names = cifdict["_entity_poly_seq.mon_id"]
    all_records = zip(all_entity_id, all_seq_id, all_res_names)
    all_res_dict = dict()
    seq_id_set = set()
    
    for entity_id, seq_id, res_name in all_records:
        
        if entity_id not in all_res_dict:
            all_res_dict[entity_id] = []
        
        # there are disordered residues that could mess up the 
        # sequence match later in constructing the chain
        # We update residue seq id to the last repeating one to avoid this
        all_res_dict[entity_id].append((seq_id, res_name))
        
    return all_res_dict

def create_missing_res_dict(cif_dict):
    missing_res_dict = dict()
    
    prefix = "_pdbx_unobs_or_zero_occ_residues"
    chain_ids = cif_dict.get(prefix+".auth_asym_id")
    res_names = cif_dict.get(prefix+".label_comp_id")
    seq_ids = cif_dict.get(prefix+".label_seq_id")
    
    if chain_ids == None:
        return missing_res_dict
    
    missing_res_data = zip(chain_ids, seq_ids, res_names)
    
    for entry in missing_res_data:
        chain_id, seq_id, res_name = entry
        if seq_id == '?':
            continue
        if res_name not in protein_letters_3to1_extended:
            continue
        if chain_id not in missing_res_dict:
            missing_res_dict[chain_id] = []
        missing_res_dict[chain_id].append((int(seq_id), res_name))
    return missing_res_dict

def check_broken_chains(pdb_ids):
    broken = []
    failed = {
            'ValueError':[],'KeyError':[],'IndexError':[],
            'Chain': [],'Other':[]
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for pdb_id in tqdm(pdb_ids):
            if not find_local_cif_path(pdb_id):
                print(pdb_id, ' not found!')
                continue
            try:
                chain, cifdict = get_first_chain_and_dict(pdb_id)
                chain_id_dict = create_entity_chain_dict(cifdict)
                missing_res_dict = create_missing_res_dict(cifdict)
                all_res_dict = create_all_res_dict(cifdict)
                
                entity_id, known_seq, can_seq = chain_id_dict[chain.id]
                
                if chain.id in missing_res_dict:
                    reported_missing_res = missing_res_dict[chain.id]
                else:
                    reported_missing_res = []
                all_res = all_res_dict[entity_id]
                
                schain = SChain(chain, known_seq, can_seq, all_res, reported_missing_res)
                if not schain.is_continuous():
                    broken.append(pdb_id)
            except ValueError:
                failed['ValueError'].append(pdb_id)
            except KeyError:
                failed['KeyError'].append(pdb_id)
            except IndexError:
                failed['IndexError'].append(pdb_id)
            except ChainConstructionException:
                failed['Chain'].append(pdb_id)
            except:
                failed['Other'].append(pdb_id)
    return broken, failed

if __name__ == "__main__":
    from random import randrange, seed
    import pickle
    import warnings
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map

    with open('/home/truman/all_pdb_proteins.txt','r') as f:
        all_pdb_ids = f.read().strip().split(',')

    # all_pdb_ids = all_pdb_ids[:1000]
    n = 500
    chunks = [all_pdb_ids[i * n:(i + 1) * n] for i in \
            range((len(all_pdb_ids) + n - 1) // n )]
    seed(42)
    with open('tqdm_progress.out', 'w') as fh:
        r = process_map(
                check_broken_chains, 
                chunks,
                file=fh,
                )
    
    all_broken = []
    all_failed = {
            'ValueError':[],'KeyError':[],'IndexError':[],
            'Chain': [],'Other':[]
    }
    for broken, failed in r:
        all_broken.extend(broken)
        for key, val in failed.items():
            all_failed[key].extend(val)
    
    # print(all_broken, all_failed)
    # broken, failed = check_broken_chains(all_pdb_ids[:500])
    

    with open('broken_pdb_chain.txt', 'w') as f:
        for pdb_id in all_broken:
            f.write(pdb_id+'\n')
    
    with open('failed_pdb_chain.txt', 'w') as f:
        for key, val in all_failed.items():
            f.write(key+'\n')
            for pdb_id in val:
                f.write('\t'+pdb_id+'\n')

    with open('failed_dict.pkl', 'wb') as p:
        pickle.dump(all_failed, p)