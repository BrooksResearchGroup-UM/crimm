from importlib.util import find_spec

if find_spec("nglview") is None:
    raise ImportError(
        "nglview not found! Install nglview to show protein structures."
        "http://nglviewer.org/nglview/latest/index.html#installation"
    )

import warnings

import nglview as nv
from Bio.PDB.Selection import unfold_entities

from crimm.IO.PDBString import get_pdb_str
from crimm.Visualization._common import rdkit_mol_to_sdf


class NGLStructure(nv.Structure):
    """NGLView Structure class for visualizing crimm.Structure.Structure on jupyter notebook"""

    def __init__(self, entity):
        super().__init__()
        self.entity = entity

    def get_structure_string(self):
        return get_pdb_str(
            self.entity,
            include_alt=False,
            trunc_resname=True,
            use_charmm_format=False,
            convert_water=True,
            chain_id_policy="renumber",
        )


class NGLRDKitStructure(nv.Structure):
    """NGLView Structure class for visualizing crimm.Structure.Structure on jupyter notebook"""

    def __init__(self, entity, conf_id=None):
        super().__init__()
        self.entity = entity
        self.ext = "sdf"
        self.conf_id = conf_id

    def get_structure_string(self):
        return rdkit_mol_to_sdf(self.entity, conf_id=self.conf_id)


def _set_size(view, width=None, height=None):
    """Size an nglview widget. Numbers are pixels, as in the py3Dmol backend."""
    if width is not None:
        view.layout.width = f"{width}px" if isinstance(width, (int, float)) else width
    if height is not None:
        view.layout.height = (
            f"{height}px" if isinstance(height, (int, float)) else height
        )
    return view


def _load_ngl_view(entity, view):
    """Load entity into nglview instance"""
    ngl_structure = NGLStructure(entity)
    component = view.add_component(ngl_structure)
    if entity.level == "C" and entity.chain_type == "Solvent":
        view.add_representation("licorice", component=component._index)


def show_nglview(entity, width=None, height=None):
    """
    Load pdb string into nglview instance
    """
    view = _set_size(nv.NGLWidget(), width, height)
    if entity.level == "S":
        for chain in entity.models[0].chains:
            _load_ngl_view(chain, view)
    elif entity.level == "M":
        for chain in entity.chains:
            _load_ngl_view(chain, view)
    else:
        _load_ngl_view(entity, view)
    return view


def show_nglview_residue(residue, width=None, height=None):
    """Show a residue in context of its polymer chain.

    The whole chain is rendered as a subdued grey cartoon and the residue is
    shown as ball+stick on top. Falls back to a plain ball+stick view when the
    residue has no parent polymer chain.
    """
    from crimm.StructEntities.Chain import PolymerChain

    view = _set_size(nv.NGLWidget(), width, height)
    chain = residue.parent
    if chain is not None and isinstance(chain, PolymerChain):
        res_id = residue.id[1]
        ngl_structure = NGLStructure(chain)
        component = view.add_component(ngl_structure)
        comp_idx = component._index
        view.clear_representations(component=comp_idx)
        view.add_representation(
            "cartoon", component=comp_idx, color="grey", opacity=0.5
        )
        view.add_representation("ball+stick", selection=str(res_id), component=comp_idx)
        view.center(selection=str(res_id))  # resid
    else:
        ngl_structure = NGLStructure(residue)
        view.add_component(ngl_structure)

    return view


def show_nglview_multiple(entity_list, width=None, height=None):
    """
    Load a list of entity into nglview instance
    """
    view = _set_size(nv.NGLWidget(), width, height)
    for entity in entity_list:
        ngl_structure = NGLStructure(entity)
        component = view.add_component(ngl_structure)
        if entity.level == "C" and entity.chain_type == "Solvent":
            view.add_representation("licorice", component=component._index)
    return view


## TODO(#74): Add feature for add_representation() directly for entity list
class View(nv.NGLWidget):
    """Customized NGLView Widget class for visualizing crimm.Structure.Structure on jupyter notebook"""

    def __init__(self, width=None, height=None):
        super().__init__()
        _set_size(self, width, height)
        self.entity_dict = {}
        self.atom_id_lookup = {}

    def _load_entity(self, entity):
        ngl_structure = NGLStructure(entity)
        component = self.add_component(ngl_structure)
        self._create_atom_lookup(entity, component._index)
        self.entity_dict[entity] = component
        return component

    def _create_atom_lookup(self, entity, component_index):
        """create a lookup table for atom id and atom object"""
        for i, atom in enumerate(entity.get_atoms()):
            self.atom_id_lookup[atom] = (component_index, i)

    def load_entity(self, entity):
        """Load entity into NGLView widget. Entity can be a Structure, Model, Chain, Residue, or Atom object."""
        if entity.level == "S":
            entities = entity.models[0].chains
        elif entity.level == "M":
            entities = entity.chains
        elif entity.level in ("C", "R", "A"):
            entities = [entity]
        components = []
        for entity in entities:
            components.append(self._load_entity(entity))
        return components

    def load_rdkit(self, mol, conf_id=None):
        """Load an RDKit molecule. All conformers are shown unless conf_id is given."""
        ngl_structure = NGLRDKitStructure(mol, conf_id=conf_id)
        component = self.add_component(ngl_structure)
        self.entity_dict[mol] = component
        return component

    def subdue_all_entities(self, color="grey"):
        """subdue the colors for all entities. Deafult color is grey."""
        for i in range(len(self.entity_dict)):
            self.update_representation(component=i, color=color, opacity=0.5)

    def _create_selected_atom_id_lookup(self, selected_atoms):
        """create a lookup table for ngl atom id and keyed by component id from
        a list of selected atom object"""
        atom_id_dict = {}
        for atom in selected_atoms:
            entity_id, atom_id = self.atom_id_lookup[atom]
            if entity_id not in atom_id_dict:
                atom_id_dict[entity_id] = []
            atom_id_dict[entity_id].append(atom_id)
        return atom_id_dict

    def highlight_residues(self, residues, add_licorice=False, color="red", **kwargs):
        """
        Highlight the repaired gaps with red color and show licorice
        representations
        """
        if len(self.entity_dict) == 0:
            raise ValueError("No entity loaded!")

        if len(residues) == 0:
            warnings.warn("No residues provided for highlighting!")
            return
        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, "C"))
        residue_chains = set(unfold_entities(residues, "C"))
        if not residue_chains.issubset(entity_chains):
            raise ValueError("Residues are not from the loaded entity!")

        # Get all atoms from the residues
        atoms = unfold_entities(residues, "A")
        if len(atoms) == 0:
            # list of empty residues
            warnings.warn("No atoms provided for highlighting!")
            return

        atom_id_dict = self._create_selected_atom_id_lookup(atoms)
        self.subdue_all_entities()

        if "colorScheme" not in kwargs:
            kwargs["color"] = color
        # highlight the residues
        for chain in residue_chains:
            component = self.entity_dict[chain]
            comp_idx = component._index
            self.clear_representations(component=comp_idx)
            self.add_representation(
                "cartoon", component=comp_idx, color="grey", opacity=0.5
            )
            atom_id_selection = atom_id_dict[comp_idx]
            # TODO(#75): Find API for changing color of specific residues
            # since add cartoon representation will not render
            self.add_representation(
                "cartoon", selection=atom_id_selection, component=comp_idx, **kwargs
            )

            if add_licorice:
                # Add licorice representations
                self.add_representation(
                    "licorice",
                    selection=atom_id_selection,
                    component=comp_idx,
                    **kwargs,
                )

    def highlight_atoms(self, atoms, add_licorice=True, color="red", **kwargs):
        """highlight atoms in the loaded structure in Viewer"""
        if len(atoms) == 0:
            warnings.warn("No atoms provided for highlighting!")
            return

        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, "C"))
        atom_chains = set(unfold_entities(atoms, "C"))
        if not atom_chains.issubset(entity_chains):
            raise ValueError("Atoms are not from the loaded entity!")

        # Organize the selected atoms ids by component id
        atom_id_dict = self._create_selected_atom_id_lookup(atoms)
        representation = "licorice" if add_licorice else "cartoon"

        self.subdue_all_entities()
        if "colorScheme" not in kwargs:
            kwargs["color"] = color
        for chain in atom_chains:
            component = self.entity_dict[chain]
            comp_idx = component._index
            # atom id is 0-based index in the chain
            atom_ids = atom_id_dict[comp_idx]
            self.clear_representations(component=comp_idx)
            self.add_representation(
                "cartoon", component=comp_idx, color="grey", opacity=0.5
            )
            self.add_representation(
                representation, selection=atom_ids, component=comp_idx, **kwargs
            )

    def highlight_chains(self, chains, color="red", **kwargs):
        """highlight chains in the loaded structure in Viewer"""
        if len(chains) == 0:
            warnings.warn("No chains provided for highlighting!")
            return

        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, "C"))
        if not set(chains).issubset(entity_chains):
            raise ValueError("Chains are not from the loaded entity!")

        self.subdue_all_entities()
        if "colorScheme" not in kwargs:
            kwargs["color"] = color
        for chain in chains:
            component = self.entity_dict[chain]
            comp_idx = component._index
            self.clear_representations(component=comp_idx)
            self.add_representation("cartoon", component=comp_idx, **kwargs)

    def add_arrow(self, start, end, color=(1, 0, 0), radius=0.5, name="arrow"):
        """Draw an arrow from start to end (coordinates in Å).

        color is an RGB triple with components in [0, 1], or a colour name.
        """
        if not isinstance(color, str):
            color = list(color)
        self.shape.add_arrow(list(start), list(end), color, radius, name)
