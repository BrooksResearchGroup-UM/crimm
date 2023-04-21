from importlib.util import find_spec
if find_spec("nglview") is None:
    raise ImportError(
        "nglview not found! Install nglview to show protein structures."
        "http://nglviewer.org/nglview/latest/index.html#installation"
    )

import warnings
from typing import List
from IPython.display import display
import nglview as nv
from Bio.PDB.Selection import unfold_entities
from crimm.IO import get_pdb_str
import crimm.StructEntities as Entities

class NGLStructure(nv.Structure):
    """NGLView Structure class for visualizing crimm.Structure.Structure on jupyter notebook"""
    def __init__(self, entity):
        super().__init__()
        self.entity = entity

    def get_structure_string(self):
        return get_pdb_str(self.entity, include_alt = False, trunc_resname=True)

def show_nglview(entity):
    """
    Load pdb string into nglview instance
    """
    view = nv.NGLWidget()
    ngl_structure = NGLStructure(entity)
    view.add_component(ngl_structure)
    display(view)

def show_nglview_multiple(entity_list):
    """
    Load pdb string into nglview instance
    """
    view = nv.NGLWidget()
    for entity in entity_list:
        ngl_structure = NGLStructure(entity)
        view.add_component(ngl_structure)
    display(view)

## TODO: Add feature for add_representation() directly for entity list
class View(nv.NGLWidget):
    """Customized NGLView Widget class for visualizing crimm.Structure.Structure on jupyter notebook"""
    def __init__(self):
        super().__init__()
        self.entity_dict = {}

    def _load_entity(self, entity):
        ngl_structure = NGLStructure(entity)
        component = self.add_component(ngl_structure)
        self.entity_dict[entity] = component
        return component
    
    def load_entity(self, entity):
        """Load entity into NGLView widget. Entity can be a Structure, Model, Chain, Residue, or Atom object."""
        if entity.level == 'S':
            entities = entity.models[0].chains
        elif entity.level == 'M':
            entities = entity.chains
        elif entity.level in ('C', 'R', 'A'):
            entities = [entity]
        components = []
        for entity in entities:
            components.append(self._load_entity(entity))
        return components

    # def load_entity(self, entity, defaultRepr=True):
        # self.entity_list.append(entity)
        # blob = get_pdb_str(entity, include_alt = False)
        # ngl_args = [{'type': 'blob', 'data': blob, 'binary': False}]
        # self._ngl_component_names.append(entity.get_id())
        # self._remote_call(
        #         "loadFile",
        #         target='Stage',
        #         args=ngl_args,
        #         kwargs= {'ext':'pdb',
        #         'defaultRepresentation': defaultRepr}
        #     )
        # self._ngl_component_ids.append(str(uuid.uuid4()))
        # self._update_component_auto_completion()
        # # entity.ngl_component_handle = self[-1]

    def subdue_all_entities(self, color = 'grey'):
        """subdue the colors for all entities. Deafult color is grey."""
        for i in range(len(self.entity_dict)):
            self.update_representation(component=i, color=color, opacity=0.5)

    def highlight_residues(
            self,
            residues: List[Entities.Residue],
            add_licorice = False,
            highlight_color = 'red'
        ):
        """
        Highlight the repaired gaps with red color and show licorice 
        representations
        """
        if len(self.entity_dict) == 0:
            raise ValueError('No entity loaded!')
        
        if len(residues) == 0:
            warnings.warn('No residues provided for highlighting!')
            return
        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, 'C'))
        residue_chains = set(unfold_entities(residues, 'C'))
        if not residue_chains.issubset(entity_chains):
            raise ValueError('Residues are not from the loaded entity!')

        # Select atoms by atom indices
        atom_id_selection = []
        for res in residues:
            # nglview uses 0-based index
            atom_ids = [atom.get_serial_number()-1 for atom in res]
            atom_id_selection.extend(atom_ids)

        if len(atom_id_selection) == 0:
            # list of empty residues
            warnings.warn('No atoms provided for highlighting!')
            return
        
        self.subdue_all_entities()
        # highlight the residues
        for chain in residue_chains:
            component = self.entity_dict[chain]
            comp_idx = component._index
            self.clear_representations(component=comp_idx)
            self.add_representation(
                'cartoon', component=comp_idx, color='grey', opacity=0.5
            )
            # TODO: Find API for changing color of specific residues
            # since add cartoon representation will not render 
            self.add_representation(
                'cartoon', selection=atom_id_selection, color=highlight_color
            )
        
        # Convert to string array for JS
        # sele_str = "@" + ",".join(str(s) for s in atom_id_selection)
        # # Highlight the residues (does not work)
        # self._remote_call("updateRepresentations",
        #                   target='compList',
        #                   kwargs={
        #                         'component_index': 0,
        #                         'sele': sele_str,
        #                         'color': highlight_color
        #                     })

        
        # self._remote_call('addRepresentation',
        #                 target='compList',
        #                 args=['cartoon'],
        #                 kwargs={
        #                     'sele': sele_str, 
        #                     'color': color, 
        #                     'component_index': 0
        #                 }
        #                 )
        if add_licorice:
            # Add licorice representations
            self.add_representation(
                'licorice', selection=atom_id_selection, color=highlight_color
            )
            
            # self._remote_call('addRepresentation',
            #                 target='compList',
            #                 args=['licorice'],
            #                 kwargs={
            #                     'sele': sele_str,  
            #                     'component_index': 0
            #                 }
            #                 )

    def highlight_atoms(
            self,
            atoms: List[Entities.Atom],
            add_licorice = True,
            highlight_color = 'red'
        ):
        """highlight atoms in the loaded structure in Viewer"""
        if len(atoms) == 0:
            warnings.warn('No atoms provided for highlighting!')
            return
        
        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, 'C'))
        atom_chains = set(unfold_entities(atoms, 'C'))
        if not atom_chains.issubset(entity_chains):
            raise ValueError('Residues are not from the loaded entity!')
        
        # nglview uses 0-based index
        atom_ids = [atom.get_serial_number()-1 for atom in atoms]
        representation = 'licorice' if add_licorice else 'cartoon'

        self.subdue_all_entities()
        for chain in atom_chains:
            component = self.entity_dict[chain]
            comp_idx = component._index
            self.clear_representations(component=comp_idx)
            self.add_representation(
                'cartoon', component=comp_idx, color='grey'
            )
            self.add_representation(
                representation, selection= atom_ids, color=highlight_color
            )

        
        # # Convert to string array for JS
        # sele_str = "@" + ",".join(str(s) for s in atom_ids)
        
        
        # # Add the highlighted representation
        
        # self._remote_call(
        #         'addRepresentation',
        #         target='compList',
        #         args=[representation],
        #         kwargs={
        #             'sele': sele_str,
        #             'color': highlight_color,
        #             'component_index': component_idx
        #         }
        #     )


