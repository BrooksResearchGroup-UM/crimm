"""This module contains the Solvator class, which solvates a Structure, Model,
or Chain level entity with water molecules.
"""
import os
import warnings
import numpy as np
from scipy.spatial import KDTree
from crimm import Data
from crimm.Modeller.CoordManipulator import CoordManipulator
from crimm.StructEntities import Atom, Residue, Chain, Model

WATER_COORD_PATH = os.path.join(os.path.dirname(Data.__file__), 'water_coords.npy')
BOXWIDTH=18.662 # water unit cube width

# Atom objects for water
OH2 = Atom(
    name = 'OH2',
    coord = None,
    bfactor = 0.0,
    occupancy = 1.0,
    altloc = ' ',
    fullname = 'OH2',
    serial_number = 0,
    element = 'O'
)

H1 = Atom(
    name = 'H1',
    coord = None,
    bfactor = 0.0,
    occupancy = 1.0,
    altloc = ' ',
    fullname = 'H1',
    serial_number = 0,
    element = 'H'
)

H2 = Atom(
    name = 'H2',
    coord = None,
    bfactor = 0.0,
    occupancy = 1.0,
    altloc = ' ',
    fullname = 'H2',
    serial_number = 0,
    element = 'H'
)

## TODO: deal with atom serial number > 99999 for larger structures (e.g. 1A8I)
class Solvator:
    """Solvates a Structure, Model, or Chain level entity with water molecules.
    The solvated entity will be returned as a Model level entity. The solvated
    entity will be centered in a cubic box with side length equal to the
    maximum dimension of the entity plus the cutoff distance. (i.e., Coordinates 
    will be oriented using CoordManipulator.orient_coords() before solvation.)
    The solvcut distance is the distance from the solute at which water
    molecules will be removed. The solvcut distance is used to remove water 
    molecules that are too close to the solute. 
    If altloc atoms exist in the entity, the first altloc atoms will be used to
    determine water molecules location during solvation.

    Parameters
    ----------
    entity : Structure, Model, or Chain level entity
        The entity to solvate. If a Structure level entity is provided, the
        first Model will be solvated. If a Model level entity is provided, all 
        chains in the model will be solvated. If a Chain level entity is 
        provided, the chain will be solvated.
    cutoff : float, optional
        The distance from the solute at which the water box's boundary extends to.
        The default is 9.0 A.
    solvcut : float, optional
        The distance from the solute at which water molecules will be removed.
        The default is 2.10 A.
    remove_existing_water : bool, optional
        If True, any existing water molecules in the entity will be removed.
        The default is True.
    
    Returns
    -------
    solvated_entity : Model level entity
        The solvated entity.
    
    Examples
    --------
    >>> from crimm import fetch_rcsb
    >>> from crimm.Modeller.Solvator import Solvator

    >>> structure = fetch_rcsb('5igv')
    >>> fisrt_model = structure.models[0]
    >>> solvator = Solvator()
    >>> solvated_model = solvator.solvate(fisrt_model)
    >>> water_chains = [
        chain for chain in solvated if chain.chain_type == 'Solvent'
    ]

    Note that water chains are named W[A-Z] and have a maximum number of 9999 residues
    >>> water_chains 
    [<Solvent id=WA Residues=9999>, <Solvent id=WB Residues=2486>]
    >>> solvator.water_box_coords.shape # shape in (N waters, 3 atoms, 3 coords)
    (12485, 3, 3)
    >>> chain = structure[1]['A'] # get chain A from the first model
    >>> solvated_chain = solvator.solvate(chain)

    More water molecules are added to solvate the chain since the ligands are
    not included in the solvation process
    >>> solvator.water_box_coords.shape 
    (12531, 3, 3)

    """

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def __init__(self) -> None:
        self.solvated_model = None
        self._cur_water_chain = None
        self._alphabet_index = None
        self.cutoff = None
        self.solvcut = None
        self.coords = None
        self.box_dim = None
        self.water_box_coords = None
        # unit of pre-equilibrated cube of water molecules (18.662 A each side)
        self.water_unit_coords = np.load(WATER_COORD_PATH)

    def solvate(
            self, entity, cutoff=9.0, solvcut = 2.10, 
            remove_existing_water = True
        ) -> Model:
        """Solvates the entity and returns a Model level entity. The solvated
        entity will be centered in a cubic box with side length equal to the
        maximum dimension of the entity plus the cutoff distance. (i.e.,
        Coordinates will be oriented using CoordManipulator.orient_coords()
        before solvation.) The solvcut distance is the distance from the solute
        at which water molecules will be removed. The solvcut distance is used
        to remove water molecules that are too close to the solute. If altloc
        atoms exist in the entity, the first altloc atoms will be used to
        determine water molecules location during solvation."""

        self.cutoff = cutoff
        self.solvcut = solvcut
        if entity.level == 'S':
            self.solvated_model = entity.models[0]
        elif entity.level == 'M':
            self.solvated_model = entity
        elif entity.level == 'C':
            self.solvated_model = Model(1)
            self.solvated_model.add(entity)
        else:
            raise ValueError(
                'Solvator can only take Structure, Model, or Chain level entities'
            )
        if remove_existing_water:
            self.remove_existing_water(self.solvated_model)
        if len(self.solvated_model.chains) == 0:
            raise ValueError('No chains in model to solvate')

        coorman = CoordManipulator()
        coorman.load_entity(entity)
        coorman.orient_coords(apply_to_parent=True)
        self.box_dim = (coorman.box_dim+self.cutoff).max() # for cubic box
        self.coords = self._extract_coords(self.solvated_model)
        self._solvate_model()
        return self.solvated_model

    def _extract_coords(self, entity) -> np.ndarray:
        """Extracts coordinates from entity. If any altloc atoms are present, 
        only the first altloc atoms will be included in the returned array."""
        coords = []
        for atom in entity.get_atoms(include_alt=False):
            coords.append(atom.get_coord())
        return np.array(coords)

    def remove_existing_water(self, model: Model) -> Model:
        """Removes existing water molecules from the model."""
        remove_list = []
        for chain in model.chains:
            if chain.chain_type == 'Solvent':
                remove_list.append(chain.id)
        for chain_id in remove_list:
            warnings.warn(
                f'Removing existing water chain {chain_id} from model',
                UserWarning
            )
            model.detach_child(chain_id)
    
    def create_water_box_coords(self) -> np.ndarray:
        """Creates a cubic box of water molecules centered at the origin (0, 0, 0)."""
        n_water_cubes = int(np.ceil(self.box_dim / BOXWIDTH))
        water_coords_expanded = self.water_unit_coords.reshape(-1,3)
        n_atoms = water_coords_expanded.shape[0]
        x_coords = water_coords_expanded[:,0]
        water_line = np.empty((n_atoms*n_water_cubes, 3)) # stride along x-axis
        water_plane = np.empty((n_atoms*n_water_cubes**2, 3)) # stride along y-axis
        water_box = np.empty((n_atoms*n_water_cubes**3, 3)) # stride along z-axis
        
        for i in range(n_water_cubes):
            st, end = i*n_atoms, (i+1)*n_atoms
            water_line[st:end,0] = x_coords+i*BOXWIDTH
            water_line[st:end,1:] = water_coords_expanded[:,1:]

        n_atoms_per_line = water_line.shape[0]
        for i in range(n_water_cubes):
            st, end = i*n_atoms_per_line, (i+1)*n_atoms_per_line
            water_plane[st:end] = water_line
            water_plane[st:end,1] += i*BOXWIDTH

        n_atoms_per_plane = water_plane.shape[0]
        for i in range(n_water_cubes):
            st, end = i*n_atoms_per_plane, (i+1)*n_atoms_per_plane
            water_box[st:end] = water_plane
            water_box[st:end,2] += i*BOXWIDTH

        # recenter the box
        translation_vec = -water_box.ptp(0)/2 - water_box.min(0)
        water_box += translation_vec

        return water_box

    def get_expelled_water_box_coords(self) -> np.ndarray:
        """Returns water molecules that are outside the solvcut distance from
        the solute."""
        water_box = self.create_water_box_coords() # (N_atoms, 3)
        c1 = water_box > self.box_dim/2
        c2 = water_box < -self.box_dim/2
        boundary_select = np.logical_not(
            np.any((c1 | c2).reshape(-1,3,3), axis=(1,2))
        ) # (N_waters,)
        
        kd_tree = KDTree(self.coords)
        water_kd_tree = KDTree(water_box)

        r = water_kd_tree.query_ball_tree(kd_tree, self.solvcut)

        within_cutoff = np.empty(len(r), dtype = bool)
        for i, nei_list in enumerate(r):
            within_cutoff[i] = bool(len(nei_list))
        cutoff_select = np.logical_not(
            np.any(within_cutoff.reshape(-1,3), axis=1)
        ) # (N_waters,)

        water_box = water_box.reshape(-1,3,3)[
            boundary_select & cutoff_select
        ] # (N_waters, 3, 3)

        return water_box

    def _create_new_water_chain(self) -> Chain:
        chain_id = 'W'+self.alphabet[self._alphabet_index]
        water_chain = Chain(chain_id)
        water_chain.chain_type = 'Solvent'
        water_chain.pdbx_description = 'water'
        return water_chain

    def _solvate_model(self):
        self.water_box_coords = self.get_expelled_water_box_coords()
        self._alphabet_index = 0
        self._cur_water_chain = self._create_new_water_chain()
        self.solvated_model.add(self._cur_water_chain)

        for i, res_coords in enumerate(self.water_box_coords):
            # split water molecules into chains of 9999 residues for PDB format
            # compliance
            resseq = i % 9999 + 1
            if i > 0 and resseq == 1:
                self._cur_water_chain.reset_atom_serial_numbers()
                self._alphabet_index += 1
                self._cur_water_chain = self._create_new_water_chain()
                self.solvated_model.add(self._cur_water_chain)

            water_res = Residue((' ', resseq, ' '), 'HOH', '')

            cur_oxygen = OH2.copy()
            cur_h1 = H1.copy()
            cur_h2 = H2.copy()

            OH2_coord, H1_coord, H2_coord = res_coords
            cur_oxygen.coord = OH2_coord
            cur_h1.coord = H1_coord
            cur_h2.coord = H2_coord

            water_res.add(cur_oxygen)
            water_res.add(cur_h1)
            water_res.add(cur_h2)
            self._cur_water_chain.add(water_res)
        self._cur_water_chain.reset_atom_serial_numbers()

