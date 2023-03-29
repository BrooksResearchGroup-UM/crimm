from importlib.util import find_spec
if find_spec("nglview") is None:
    raise ImportError(
        "nglview not found! Install nglview to show protein structures."
        "http://nglviewer.org/nglview/latest/index.html#installation"
    )
import warnings
import nglview as nv

class StructureIO:
    """
    Class to handle input and output of Structure objects for visualization
    """
    def __init__(self, structure):
        self.structure = structure
    
    def set_structure(self, entity, include_alt = True, reset_serial = True):
        self.entity = entity.copy()
        if reset_serial:
            self.entity.reset_atom_serial_numbers(include_alt=include_alt)


def _load_entity_in_view(
        entity,
        view: nv.NGLWidget,
        defaultRepr,
    ):
    blob = entity.get_pdb_str(include_alt = False)
    ngl_args = [{'type': 'blob', 'data': blob, 'binary': False}]
    view._ngl_component_names.append(entity.get_id())
    view._remote_call(
            "loadFile",
            target='Stage',
            args=ngl_args,
            kwargs= {'ext':'pdb',
            'defaultRepresentation': defaultRepr}
        )
    
def load_nglview(entity, defaultRepr = True):
    """
    Load pdb string into nglview instance
    """
    view = nv.NGLWidget()
    _load_entity_in_view(entity, view, defaultRepr)
    # view.add_licorice('not protein')
    return view

def load_nglview_multiple(entity_list, defaultRepr = True):
    """
    Load pdb string into nglview instance
    """
    view = nv.NGLWidget()
    for entity in entity_list:
        _load_entity_in_view(entity, view, defaultRepr)
    # view.add_licorice('not protein')
    return view

def highlight_residues(
        chain, 
        residues = [], 
        res_seq_ids = [], 
        add_licorice = False
    ):
    """
    Highlight the repaired gaps with red color and show licorice 
    representations
    """
    ## FIXME: refactor these codes
    if len(res_seq_ids) > 0:
        residues = []
        for res in chain:
            if res.id[1] in res_seq_ids:
                residues.append(res)
    
    if len(residues) == 0:
        raise ValueError('List of Residues or Residue Sequence ID is required')
    
    view = nv.NGLWidget()
    blob = chain.get_pdb_str(include_alt = False)
    ngl_args = [{'type': 'blob', 'data': blob, 'binary': False}]
    view._ngl_component_names.append('Model Chain')
    # Load data, and do not add any representation
    view._remote_call("loadFile",
                    target='Stage',
                    args=ngl_args,
                    kwargs= {'ext':'pdb',
                    'defaultRepresentation':False}
                    )
    # Add color existing residues grey
    view._remote_call('addRepresentation',
                    target='compList',
                    args=['cartoon'],
                    kwargs={
                        'sele': 'protein',
                        'color': 'grey',
                        'component_index': 0
                    }
                    )

    # Select atoms by atom indices
    res_atom_selection = []
    for res in residues:
        for res_id in res:
            # nglview uses 0-based index
            atom_ids = [atom.get_serial_number()-1 for atom in chain[res_id]]
            res_atom_selection.extend(atom_ids)

    if len(res_atom_selection) == 0:
        warnings.warn('No atoms provided for highlighting!')

    # Convert to string array for JS
    sele_str = "@" + ",".join(str(s) for s in res_atom_selection)
    # Highlight the repaired gap atoms with red color
    view._remote_call('addRepresentation',
                    target='compList',
                    args=['cartoon'],
                    kwargs={
                        'sele': sele_str, 
                        'color': 'red', 
                        'component_index': 0
                    }
                    )
    if add_licorice:
        # Add licorice representations
        view._remote_call('addRepresentation',
                        target='compList',
                        args=['licorice'],
                        kwargs={
                            'sele': sele_str,  
                            'component_index': 0
                        }
                        )
    view.center()
    return view

def highlight_atoms(
        view, 
        atom_list, 
        component_idx, 
        add_licorice = True, 
        highlight_color = 'red'
    ):
    if len(atom_list) == 0:
        warnings.warn('No atoms provided for highlighting!')
        return
    
    # nglview uses 0-based index
    atom_ids = [atom.get_serial_number()-1 for atom in atom_list] 
    # Convert to string array for JS
    sele_str = "@" + ",".join(str(s) for s in atom_ids)
    
    representation = 'licorice' if add_licorice else 'cartoon'
    # Add the highlighted representation
    
    view._remote_call(
            'addRepresentation',
            target='compList',
            args=[representation],
            kwargs={
                'sele': sele_str,
                'color': highlight_color,
                'component_index': component_idx
            }
        )
    view.center()