from importlib.util import find_spec
if find_spec("py3Dmol") is None:
    raise ImportError(
        "py3Dmol not found! Install py3Dmol to show structures."
        "https://pypi.org/project/py3Dmol/"
    )

import warnings
import py3Dmol
from Bio.PDB.Selection import unfold_entities
from crimm.IO.PDBString import get_pdb_str
from crimm.StructEntities.Chain import PolymerChain


def _get_pdb_str(entity):
    return get_pdb_str(
        entity, include_alt=False, trunc_resname=True,
        use_charmm_format=False, convert_water=True,
        chain_id_policy='renumber'
    )


def _is_polymer(entity):
    """Return True for polymer chains (protein/nucleic acid) that support cartoon."""
    return isinstance(entity, PolymerChain)


def _apply_default_style(view, model_idx, is_polymer):
    if is_polymer:
        view.setStyle({'model': model_idx}, {'cartoon': {'color': 'spectrum'}})
    else:
        view.setStyle(
            {'model': model_idx},
            {'stick': {}, 'sphere': {'radius': 0.3}}
        )


def _add_model(entity, view, model_idx):
    pdb_str = _get_pdb_str(entity)
    view.addModel(pdb_str, 'pdb')
    _apply_default_style(view, model_idx, _is_polymer(entity))


def show_py3dmol(entity, width=800, height=400):
    """Load entity into a py3Dmol view and return it."""
    view = py3Dmol.view(width=width, height=height)
    model_idx = 0
    if entity.level == 'S':
        for chain in entity.models[0].chains:
            _add_model(chain, view, model_idx)
            model_idx += 1
    elif entity.level == 'M':
        for chain in entity.chains:
            _add_model(chain, view, model_idx)
            model_idx += 1
    else:
        _add_model(entity, view, model_idx)
    view.zoomTo()
    return view


def show_py3dmol_multiple(entity_list, width=800, height=400):
    """Load a list of entities into a py3Dmol view and return it."""
    view = py3Dmol.view(width=width, height=height)
    for i, entity in enumerate(entity_list):
        _add_model(entity, view, i)
    view.zoomTo()
    return view


def show_py3dmol_residue(residue, width=800, height=400):
    """Show a residue in context of its polymer chain.

    The whole chain is rendered as a subdued grey cartoon and the residue is
    shown as ball+stick on top, zoomed in. Falls back to a plain ball+stick
    view when the residue has no parent polymer chain.
    """
    view = py3Dmol.view(width=width, height=height)
    chain = residue.parent
    if chain is not None and isinstance(chain, PolymerChain):
        pdb_str = _get_pdb_str(chain)
        view.addModel(pdb_str, 'pdb')
        resi = residue.id[1]
        view.setStyle({}, {'cartoon': {'color': 'grey', 'opacity': 0.5}})
        view.addStyle({'resi': resi}, {'stick': {}, 'sphere': {'radius': 0.3}})
        view.zoomTo({'resi': resi})
    else:
        pdb_str = _get_pdb_str(residue)
        view.addModel(pdb_str, 'pdb')
        view.setStyle({}, {'stick': {}, 'sphere': {'radius': 0.3}})
        view.zoomTo()
    return view


class View:
    """py3Dmol-based viewer with the same API as NGLVisualization.View."""

    def __init__(self, width=800, height=400):
        self.view = py3Dmol.view(width=width, height=height)
        self.entity_dict = {}       # entity -> model_idx
        self.atom_id_lookup = {}    # atom -> (model_idx, serial_in_pdb)
        self._polymer_models = set()  # model indices that are polymer chains
        self._model_count = 0

    def _load_entity(self, entity):
        pdb_str = _get_pdb_str(entity)
        self.view.addModel(pdb_str, 'pdb')
        model_idx = self._model_count
        self._model_count += 1

        # Build atom -> (model_idx, 1-based serial) map.
        # get_atoms() defaults to include_alt=False, matching the PDB writer.
        for i, atom in enumerate(entity.get_atoms()):
            self.atom_id_lookup[atom] = (model_idx, i + 1)

        self.entity_dict[entity] = model_idx

        is_polymer = _is_polymer(entity)
        if is_polymer:
            self._polymer_models.add(model_idx)
        _apply_default_style(self.view, model_idx, is_polymer)

        return model_idx

    def load_entity(self, entity):
        """Load entity into the viewer. Accepts Structure, Model, Chain, Residue, or Atom."""
        if entity.level == 'S':
            entities = entity.models[0].chains
        elif entity.level == 'M':
            entities = entity.chains
        elif entity.level in ('C', 'R', 'A'):
            entities = [entity]
        model_indices = [self._load_entity(e) for e in entities]
        self.view.zoomTo()
        return model_indices

    def load_rdkit(self, mol, conf_id=None):
        """Load an RDKit molecule into the viewer."""
        from rdkit import Chem
        n_conf = mol.GetNumConformers()
        if n_conf == 0:
            sdf_str = Chem.MolToMolBlock(mol)
        elif conf_id is not None:
            sdf_str = Chem.MolToMolBlock(mol, confId=conf_id)
        else:
            sdf_str = Chem.MolToMolBlock(mol, confId=0)
        self.view.addModel(sdf_str, 'sdf')
        model_idx = self._model_count
        self._model_count += 1
        self.entity_dict[mol] = model_idx
        self.view.setStyle(
            {'model': model_idx},
            {'stick': {}, 'sphere': {'radius': 0.3}}
        )
        self.view.zoomTo()
        return model_idx

    def subdue_all_entities(self, color='grey'):
        """Set all loaded entities to a subdued style."""
        for model_idx in self.entity_dict.values():
            if model_idx in self._polymer_models:
                self.view.setStyle(
                    {'model': model_idx},
                    {'cartoon': {'color': color, 'opacity': 0.5}}
                )
            else:
                self.view.setStyle(
                    {'model': model_idx},
                    {
                        'stick': {'color': color, 'opacity': 0.5},
                        'sphere': {'color': color, 'radius': 0.3, 'opacity': 0.5},
                    }
                )

    def _residue_lookup_by_model(self, residues):
        """Return {model_idx: [resi, ...]} for the given residues."""
        result = {}
        for residue in residues:
            chain = residue.parent
            if chain not in self.entity_dict:
                raise ValueError(f'Chain {chain.get_id()!r} is not loaded in this viewer.')
            model_idx = self.entity_dict[chain]
            result.setdefault(model_idx, []).append(residue.id[1])
        return result

    def _atom_serial_lookup_by_model(self, atoms):
        """Return {model_idx: [serial, ...]} for the given atoms."""
        result = {}
        for atom in atoms:
            if atom not in self.atom_id_lookup:
                raise ValueError(f'Atom {atom!r} was not found in the atom lookup.')
            model_idx, serial = self.atom_id_lookup[atom]
            result.setdefault(model_idx, []).append(serial)
        return result

    def highlight_residues(self, residues, add_licorice=False, color='red', **_kwargs):
        """Highlight residues; subdues everything else."""
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

        atoms = unfold_entities(residues, 'A')
        if len(atoms) == 0:
            warnings.warn('No atoms found in the provided residues!')
            return

        self.subdue_all_entities()
        resi_by_model = self._residue_lookup_by_model(residues)
        for model_idx, resi_list in resi_by_model.items():
            sel = {'model': model_idx, 'resi': resi_list}
            self.view.addStyle(sel, {'cartoon': {'color': color}})
            if add_licorice:
                self.view.addStyle(sel, {'stick': {'color': color}})

    def highlight_atoms(self, atoms, add_licorice=True, color='red', **_kwargs):
        """Highlight atoms; subdues everything else."""
        if len(atoms) == 0:
            warnings.warn('No atoms provided for highlighting!')
            return

        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, 'C'))
        atom_chains = set(unfold_entities(atoms, 'C'))
        if not atom_chains.issubset(entity_chains):
            raise ValueError('Atoms are not from the loaded entity!')

        self.subdue_all_entities()
        serial_by_model = self._atom_serial_lookup_by_model(atoms)
        representation = 'stick' if add_licorice else 'cartoon'
        for model_idx, serial_list in serial_by_model.items():
            self.view.addStyle(
                {'model': model_idx, 'serial': serial_list},
                {representation: {'color': color}}
            )

    def highlight_chains(self, chains, color='red', **_kwargs):
        """Highlight entire chains; subdues everything else."""
        if len(chains) == 0:
            warnings.warn('No chains provided for highlighting!')
            return

        loaded_entities = list(self.entity_dict.keys())
        entity_chains = set(unfold_entities(loaded_entities, 'C'))
        if not set(chains).issubset(entity_chains):
            raise ValueError('Chains are not from the loaded entity!')

        self.subdue_all_entities()
        for chain in chains:
            model_idx = self.entity_dict[chain]
            if model_idx in self._polymer_models:
                self.view.setStyle(
                    {'model': model_idx},
                    {'cartoon': {'color': color}}
                )
            else:
                self.view.setStyle(
                    {'model': model_idx},
                    {'stick': {'color': color}, 'sphere': {'color': color, 'radius': 0.3}}
                )

    def show(self):
        """Render and display the viewer."""
        self.view.zoomTo()
        return self.view.show()

    def _repr_html_(self):
        self.view.zoomTo()
        return self.view._repr_html_()
