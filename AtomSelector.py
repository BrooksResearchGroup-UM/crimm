import numpy as np
from numpy import linalg as LA
from Bio.PDB.Atom import Atom
from Bio.PDB.Entity import Entity
from NGLVisualization import load_nglview_multiple, highlight_atoms

class AtomSelector:
    def __init__(self) -> None:
        self.cur_selections = None
        self.ref_entity = None
        self.select_entity = None
        self.point_coord = None
    
    def _check_entity(self, entity):
        if not isinstance(entity, Entity):
            raise TypeError('BioPython Entities are required as function arguments')
        entity.reset_atom_serial_numbers()
        return entity
    
    def _get_entity_coords(self, entity):
        if isinstance(entity, Atom):
            return entity.coord
        return np.asarray(
                [a.coord for a in entity.get_unpacked_atoms()]
            )
    
    def _select_by_point(self, origin, cand_coords, radius):
        diff = cand_coords - origin
        dists = LA.norm(diff, axis=1)
        mask = (dists <= radius) 
        return mask
    
    def _select_around(self, ref_coords, cand_coords, radius):
        mask = np.zeros(cand_coords.shape[0], dtype=bool)
        for coord in ref_coords:
            diffs = cand_coords - coord
            dists = LA.norm(diffs, axis=1)
            mask |= (dists <= radius)
        return mask
    
    def _get_atoms_from_mask(self, mask):
        atom_array = np.asarray(list(self.select_entity.get_unpacked_atoms()))
        self.cur_selections = list(atom_array[mask])
        return self.cur_selections
        
    def select_by_point(self, point_coord, select_entity, radius):
        self.ref_entity = None
        self.select_entity = self._check_entity(select_entity)
        self.point_coord = np.asarray(point_coord, dtype=float)

        cand_coords = self._get_entity_coords(select_entity)
        mask = self._select_by_point(self, point_coord, cand_coords, radius)
        
        return self._get_atoms_from_mask(mask)

    def select_around(self, ref_entity, select_entity, radius):
        self.point_coord = None
        self.ref_entity = self._check_entity(ref_entity)
        self.select_entity = self._check_entity(select_entity)

        cand_coords = self._get_entity_coords(select_entity)
        ref_coords = self._get_entity_coords(ref_entity)

        if isinstance(ref_entity, Atom):
            mask = self._select_by_point(ref_coords, cand_coords, radius)
        else:
            mask = self._select_around(ref_coords, cand_coords, radius)

        return self._get_atoms_from_mask(mask)
    
    def show_selection(self, show_as_licorice = True, highlight_color = 'red'):
        
        view = load_nglview_multiple(
                [self.ref_entity, self.select_entity],
                defaultRepr = False,
                reset_serial = False
            )
        
        # Set select_entity grey
        view._remote_call('addRepresentation',
                    target='compList',
                    args=['licorice'],
                    kwargs={
                        'sele': 'all',
                        'color': 'grey',
                        'component_index': 1
                    }
                    )
        # Set ref_entity cyan
        view._remote_call('addRepresentation',
                    target='compList',
                    args=['licorice'],
                    kwargs={
                        'sele': 'all',
                        'color': 'cyan',
                        'component_index': 0
                    }
                    )
        
        highlight_atoms(
                view,
                atom_list = self.cur_selections,
                component_idx = 1,
                add_licorice = show_as_licorice,
                highlight_color = highlight_color
            )
        return view

    
    
    
