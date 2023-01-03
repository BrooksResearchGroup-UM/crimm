from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from collections import namedtuple
import numpy

class ChMMCIF2Dict(dict):

    def __init__(self, filename):
        orig_dict = MMCIF2Dict(filename)
        self._organize_mmcif_dict(orig_dict)

    def _convert_type(self, entry: str):
        sign = -1 if entry[0] == '-' else 1
        entry = entry.lstrip('-')
        if entry.isnumeric():
            return sign*int(entry)
        elif entry.count('.') == 1:
            a, b = entry.split('.')
            if a.isnumeric() and b.isnumeric():
                return sign*float(entry)
        return entry
    
    def _reformat_str(self, entry: str):
        if entry in ('?','.'):
            return None
        elif '\n' in entry:
            return ''.join(entry.split('\n'))
        return self._convert_type(entry)
    
    def _organize_mmcif_dict(self, orig_dict):
        for k, v in orig_dict.items():
            k = k.strip('_')
            if isinstance(v, list):
                v = [self._reformat_str(x) for x in v]
            if '[' in k:
                k = ''.join([s.strip(']') for s in k.split('[')])
            if '-' in k:
                k = '_'.join([s for s in k.split('-')])
            if '.' not in k:
                # second level does not exist
                self[k] = v
                continue
            main, sec = k.split('.')
            if main not in self:
                self[main] = dict()
            self[main][sec] = v

    def level_two_get(self, key, subkey):
        if key in self:
            return self[key].get(subkey)
        return None
    
    def retrieve_single_value_dict(self, key):
        citation = dict()
        subdict = self.get(key)
        if subdict is None:
            return None
        for k, v in subdict.items():
            if len(v) != 1:
                raise ValueError(
                    'Sub-dict "{}" is not a single value dict'.format(key)
                )
            if v[0] is not None: 
                citation[k] = v[0]
        return citation
    
    def create_namedtuples(self, key, single_value = False):
        '''Create a list of namedtuples from a section of the mmcif dict'''
        if key not in self:
            return ()
        sub_dict = self[key]
        named_entries = namedtuple(key, sub_dict.keys())

        if single_value:
            entry = list(zip(*sub_dict.values()))[0]
            return named_entries(*entry)
        
        all_entries = []
        for entry in zip(*sub_dict.values()):
            all_entries.append(named_entries(*entry))
        return all_entries
    
    def find_atom_coords(self):
        sub_dict = self['atom_site']
        n_atoms = len(sub_dict['label_atom_id'])
        
        coords = numpy.empty((n_atoms, 3),'f') 
        for i,k in enumerate(["Cartn_x", "Cartn_y", "Cartn_z"]):
            coords[:,i] = numpy.asarray(sub_dict[k],'f')
        return coords

    def create_chain_info_dict(self):
        zero_occupancy_residues = self.create_namedtuples(
            'pdbx_unobs_or_zero_occ_residues'
        )
        missing_res_dict = dict()
        for res in zero_occupancy_residues:
            if res.auth_asym_id not in missing_res_dict:
                missing_res_dict[res.auth_asym_id] = []
            missing_res_dict[res.auth_asym_id].append(
                (res.label_seq_id, res.label_comp_id)
            )

        entity_poly = self.create_namedtuples('entity_poly')
        entity_poly_dict = dict()
        for entity in entity_poly:
            auth_chain_ids = entity.pdbx_strand_id.split(',')
            for chain_id in auth_chain_ids:
                entity_poly_dict[chain_id] = entity

        entity_poly_seq = self.create_namedtuples('entity_poly_seq')
        reported_res_dict = dict()
        for res in entity_poly_seq:
            if res.entity_id not in reported_res_dict:
                reported_res_dict[res.entity_id] = []
            reported_res_dict[res.entity_id].append((res.num, res.mon_id))
        ## TODO: Initiate Chain class directly here
        chain_info = namedtuple(
            'chain_info',
            [
                'entity_id', 
                'chain_type', 
                'known_sequence', 
                'canon_sequence', 
                'reported_res', 
                'reported_missing_res'
            ]
        )
        chain_dict = dict()
        for chain_id, entity_info in entity_poly_dict.items():
            chain_dict[chain_id] = chain_info._make([
                entity_info.entity_id,
                entity_info.type,
                entity_info.pdbx_seq_one_letter_code,
                entity_info.pdbx_seq_one_letter_code_can,
                reported_res_dict.get(entity_info.entity_id),
                missing_res_dict.get(chain_id)
            ])
        return chain_dict
