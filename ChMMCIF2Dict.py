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
    
    def _check_unassigned(self, entry):
        if entry in ('?','.'):
            return None
        return self._convert_type(entry)
    
    def _organize_mmcif_dict(self, orig_dict):
        for k, v in orig_dict.items():
            k = k.strip('_')
            if isinstance(v, list):
                v = [self._check_unassigned(x) for x in v]
                if len(v) == 1:
                    v = v[0]
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
    
    def create_namedtuple(self, key):
        if key not in self:
            return ()
        sub_dict = self[key]
        named_entries = namedtuple(key, sub_dict.keys())

        for val in sub_dict.values():
            if not hasattr(val ,'__iter__'):
                return named_entries(*sub_dict.values())

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