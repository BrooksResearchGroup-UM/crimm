from collections import namedtuple
import numpy
from Bio.PDB.MMCIF2Dict import MMCIF2Dict as _MMCIF2Dict

class MMCIF2Dict(dict):
    """A dictionary-like object that reads a mmCIF file and stores the data in a dictionary."""
    def __init__(self, filename):
        orig_dict = _MMCIF2Dict(filename)
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
                k = ''.join([s.replace(']','') for s in k.split('[')])
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