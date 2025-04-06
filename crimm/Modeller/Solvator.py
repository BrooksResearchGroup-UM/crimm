"""This module contains the Solvator class, which solvates a Structure, Model,
or Chain level entity with water molecules.
"""
import os
import warnings
import numpy as np
import math
from random import choices
from scipy.spatial import KDTree
from crimm import Data
from crimm.Modeller.CoordManipulator import CoordManipulator
from crimm.StructEntities import Atom, Residue, Model
from crimm.StructEntities.Chain import Solvent, Ion, PolymerChain
from crimm.Modeller.TopoLoader import ResidueTopologySet
from crimm.Utils.StructureUtils import get_charges
from crimm.Data.components_dict import CHARMM_PDB_ION_NAMES

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
        provided, the chain will be solvated. The entity is modified in place.

    Examples
    --------
    >>> from crimm import fetch_rcsb
    >>> from crimm.Modeller.Solvator import Solvator

    >>> fisrt_model = fetch_rcsb('5igv')
    >>> solvator = Solvator()
    >>> solvated_model = solvator.solvate(
            fisrt_model, cutoff=8.0, solvcut=2.1, remove_existing_water=True
        )
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
    available_ions = CHARMM_PDB_ION_NAMES
    def __init__(self, entity) -> None:
        if entity.level == 'S':
            self.model = entity.models[0]
        elif entity.level == 'M':
            self.model = entity
        elif entity.level == 'C':
            self.model = Model(1)
            self.model.add(entity)
        else:
            raise ValueError(
                'Solvator can only take Structure, Model, or Chain level entities'
            )
        
        self.cutoff = None
        self.solvcut = None
        self.coords = None
        self.box_dim = None
        self.water_box_coords = None
        # unit of pre-equilibrated cube of water molecules (18.662 A each side)
        self.water_unit_coords = np.load(WATER_COORD_PATH)
        self._topo_loader = self.model.topology_loader
        self.box_type = None
        self.orient_method = None
        
    def get_model(self):
        return self.model

    def solvate(
            self, cutoff=9.0, solvcut = 2.10,
            remove_existing_water = True,
            orient_coords = True,
            box_type = 'cube',
            orient_method = None

        ) -> Model:
        """Solvates the entity and returns a Model level entity. The solvated
        entity will be centered in a cubic box with side length equal to the
        maximum dimension of the entity plus the cutoff distance. (i.e.,
        Coordinates will be oriented using CoordManipulator.orient_coords()
        before solvation.) The solvcut distance is the distance from the solute
        at which water molecules will be removed. The solvcut distance is used
        to remove water molecules that are too close to the solute. If altloc
        atoms exist in the entity, the first altloc atoms will be used to
        determine water molecules location during solvation.
        
        Parameters
        ----------
        cutoff : float, optional
            The distance from the solute to the edge of the cubic box. The
            default is 9.0.
        solvcut : float, optional
            The distance from the solute at which water molecules will be
            removed. The default is 2.10.
        remove_existing_water : bool, optional
            If True, remove existing water molecules from the entity. The default
            is True.
        box_type : str, optional
            The shape of the water box. The default is'cube' (default) or 'octa' 
            to choose the water box shape.
        orient_method : str, optional
            The method to orient the coordinates before solvation. The 'default'
            (default) uses the usual orientation; octa' uses an alternative 
            orientation aiming to minimize the octahedral box.
        """

        self.cutoff = cutoff
        self.solvcut = solvcut
        self.box_type = box_type
        self.orient_method = orient_method
        if self.orient_method is None and self.box_type == "octa": 
            self.orient_method = "octa"
        else:
            self.orient_method = "default"
        
        if remove_existing_water:
            self.remove_existing_water(self.model)
        if len(self.model.chains) == 0:
            raise ValueError('No chains in model to solvate')
        if orient_coords:
            coorman = CoordManipulator()
            coorman.load_entity(self.model)
            if self.orient_method == "octa":
                coorman.orient_coords_octa(apply_to_parent=(self.model.parent is not None))
                warnings.warn("Using octahedral orientation for solvation.", UserWarning)
            else:
                coorman.orient_coords(apply_to_parent=(self.model.parent is not None))
                warnings.warn("Using default orientation for solvation.", UserWarning)

            warnings.warn(
                'Orienting coordinates before solvation. This may change the '
                'atom coordinates of the entity in the structure.',
                UserWarning
            )
        self.coords = self._extract_coords(self.model)
        self.box_dim = (self.coords.ptp(0)+self.cutoff).max()

        return self._solvate_model()


    def _extract_coords(self, entity) -> np.ndarray:
        """Extracts coordinates from entity. If any altloc atoms are present, 
        only the first altloc atoms will be included in the returned array."""
        coords = []
        for atom in entity.get_atoms(include_alt=False):
            coords.append(atom.coord)
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
        """
        Creates a water box grid based on the chosen box type.
          - For 'cube': builds a cubic grid with side length = box_dim.
          - For 'octa': builds a grid over a cube of side length = box_dim * sqrt(4/3)
            (the bounding cube of a truncated octahedron) and then selects only those
            water molecules whose oxygen atom is at least 'solvcut' inside the octahedron.
        """
        if self.box_type == "cube":
            n_water_cubes = int(np.ceil(self.box_dim / BOXWIDTH))
            water_coords_expanded = self.water_unit_coords.reshape(-1, 3)
            n_atoms = water_coords_expanded.shape[0]
            water_line = np.empty((n_atoms * n_water_cubes, 3))
            water_plane = np.empty((n_atoms * n_water_cubes ** 2, 3))
            water_box = np.empty((n_atoms * n_water_cubes ** 3, 3))
            
            for i in range(n_water_cubes):
                st, end = i * n_atoms, (i + 1) * n_atoms
                water_line[st:end, 0] = water_coords_expanded[:, 0] + i * BOXWIDTH
                water_line[st:end, 1:] = water_coords_expanded[:, 1:]
            
            n_atoms_per_line = water_line.shape[0]
            for i in range(n_water_cubes):
                st, end = i * n_atoms_per_line, (i + 1) * n_atoms_per_line
                water_plane[st:end] = water_line
                water_plane[st:end, 1] += i * BOXWIDTH
            
            n_atoms_per_plane = water_plane.shape[0]
            for i in range(n_water_cubes):
                st, end = i * n_atoms_per_plane, (i + 1) * n_atoms_per_plane
                water_box[st:end] = water_plane
                water_box[st:end, 2] += i * BOXWIDTH
            
            # Recenter the box
            translation_vec = -water_box.ptp(0) / 2 - water_box.min(0)
            water_box += translation_vec
            return water_box
        elif self.box_type == "octa":
            # Bounding cube side length for the octahedron is box_dim * sqrt(4/3)
            grid_length = self.box_dim * math.sqrt(4 / 3)
            n_units = int(math.ceil(grid_length / BOXWIDTH))
            water_coords_expanded = self.water_unit_coords.reshape(-1, 3)
            n_atoms = water_coords_expanded.shape[0]
            water_points = []
            for i in range(n_units):
                for j in range(n_units):
                    for k in range(n_units):
                        translation = np.array([i * BOXWIDTH, j * BOXWIDTH, k * BOXWIDTH])
                        for atom in water_coords_expanded:
                            water_points.append(atom + translation)
            water_points = np.array(water_points)
            # Recenter the grid so that its bounding box is centered at the origin
            translation_vec = -water_points.ptp(0) / 2 - water_points.min(0)
            water_points += translation_vec
            # Reshape into water molecules (each with 3 atoms)
            water_box = water_points.reshape(-1, 3, 3)
            # Filter water molecules by testing that the oxygen atom (first atom)
            # is at least 'solvcut' inside the truncated octahedron.
            selected_waters = []
            for water in water_box:
                oxygen = water[0]
                x, y, z = oxygen
                d = self.box_dim / math.sqrt(3)
                # Compute differences for the square faces:
                dist1 = abs(x) - d
                dist2 = abs(y) - d
                dist3 = abs(z) - d
                # Compute differences for the hexagonal faces:
                dist4 = (abs(x + y + z) - self.box_dim) / math.sqrt(3)
                dist5 = (abs(x + y - z) - self.box_dim) / math.sqrt(3)
                dist6 = (abs(x - y + z) - self.box_dim) / math.sqrt(3)
                dist7 = (abs(x - y - z) - self.box_dim) / math.sqrt(3)
                sdf_value = max(dist1, dist2, dist3, dist4, dist5, dist6, dist7)
                if sdf_value <= -self.solvcut:
                    selected_waters.append(water)
            return np.array(selected_waters)
        else:
            raise ValueError("Unsupported box type")

    def get_expelled_water_box_coords(self) -> np.ndarray:
        """
        Returns water molecules that are outside the solvcut distance from the solute.
        For the cubic box the original boundary selection is applied; for the
        octahedral box, the grid (already filtered by the truncated octahedron
        condition) is further filtered by ensuring that water oxygens are not within
        solvcut of the solute.
        """
        if self.box_type == "cube":
            water_box = self.create_water_box_coords()  # shape (N_points, 3)
            # Select water molecules fully within the cubic boundary.
            c1 = water_box > self.box_dim / 2
            c2 = water_box < -self.box_dim / 2
            boundary_select = np.logical_not(
                np.any((c1 | c2).reshape(-1, 3, 3), axis=(1, 2))
            )
            kd_tree = KDTree(self.coords)
            water_kd_tree = KDTree(water_box)
            r = water_kd_tree.query_ball_tree(kd_tree, self.solvcut)
            within_cutoff = np.empty(len(r), dtype=bool)
            for i, nei_list in enumerate(r):
                within_cutoff[i] = bool(len(nei_list))
            cutoff_select = np.logical_not(
                np.any(within_cutoff.reshape(-1, 3), axis=1)
            )
            water_box = water_box.reshape((-1, 3, 3))[boundary_select & cutoff_select]
            return water_box
        elif self.box_type == "octa":
            water_box = self.create_water_box_coords()  # Already shape (N_waters, 3, 3)
            # Use the oxygen atom (first atom) of each water for KDTree filtering.
            oxy_coords = np.array([water[0] for water in water_box])
            kd_tree = KDTree(self.coords)
            water_kd_tree = KDTree(oxy_coords)
            r = water_kd_tree.query_ball_tree(kd_tree, self.solvcut)
            cutoff_select = []
            for nei_list in r:
                cutoff_select.append(not bool(len(nei_list)))
            cutoff_select = np.array(cutoff_select)
            water_box = water_box[cutoff_select]
            return water_box
        else:
            raise ValueError("Unsupported box type")

    def _create_new_water_chain(self, alphabet_index) -> Solvent:
        chain_id = 'W'+self.alphabet[alphabet_index]
        water_chain = Solvent(chain_id)
        water_chain.pdbx_description = 'water'
        water_chain.source = 'generated'
        return water_chain

    def _solvate_model(self):
        self.water_box_coords = self.get_expelled_water_box_coords()
        alphabet_index = 0
        assert self.water_box_coords.shape[1:] == (3, 3), \
        f'Invalid water box coords shape {self.water_box_coords.shape}'
        water_chains = []
        for i, res_coords in enumerate(self.water_box_coords):
            # split water molecules into chains of 9999 residues for PDB format
            # compliance
            resseq = i % 9999 + 1
            if resseq == 1:
                cur_water_chain = self._create_new_water_chain(alphabet_index)
                alphabet_index += 1
                water_chains.append(cur_water_chain)

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
            cur_water_chain.add(water_res)

        for water_chain in water_chains:
            self.model.add(water_chain)
            if self._topo_loader is not None:
                self._topo_loader.generate_solvent(
                    water_chain, solvent_model='TIP3'
                )

        return water_chains

            
    def add_balancing_ions(
            self, present_charge = None, cation='SOD', anion='CLA', skip_undefined=True
        ) -> Ion:
        """Add balancing ions to the solvated entity to bring total charge to zero.
        The default cation is Na+ and the default anion is Cl-. If the entity is
        not a solvated entity, a ValueError will be raised. A random selection of
        water molecules in the water box will be replaced with balancing ions.
        Returns a chain containing the balancing ions.
        
        Parameters
        ----------
        entity : Structure, Model, or Chain level entity
            The solvated entity to add balancing ions to.
        present_charge : int, optional
            The present charge of the solvated entity. If None, the charge will be
            calculated from the entity. The default is None. If for any reason you
            want to balance the charge to a non-zero value, you can specify it here.
        cation : str, optional
            The cation to use. The default is 'SOD' (Na+).
        anion : str, optional
            The anion to use. The default is 'CLA' (Cl-).
                
        Returns
        -------
        ion_chain : Chain
            A chain containing the balancing ions.
        """
        solvents = [chain for chain in self.model if chain.chain_type == 'Solvent']
        if len(solvents) == 0:
            raise ValueError(
                'Entity must be a solvated Structure or Model'
            )
        
        if present_charge is None:
            charge_dict = {}
            total_charges = 0
            for chain in self.model:
                if chain.chain_type == 'Solvent':
                    continue
                if chain.total_charge is None:
                    if not skip_undefined:
                        raise ValueError(
                            'Chain {chain.id} has no topology definition for atom charge! '
                            'Cannot calculate total charge.'
                        )
                    
                    warnings.warn(
                        'Chain {chain.id} has no topology definition for atom charge! '
                        'Assume zero charge.',
                    )
                charge_dict[chain.id] = chain.total_charge
                total_charges+=chain.total_charge
            print(f'Total charges before adding ions: {total_charges}')
            for chain_id, charge in charge_dict.items():
                print(f'  [Chain {chain_id}] {charge}')
        else:
            total_charges = present_charge
            print(f'Total charges before adding ions: {total_charges}')

        if total_charges == 0:
            warnings.warn('No balancing ions needed', UserWarning)
            return None
        
        if abs(int(total_charges)-total_charges) > 1e-2:
            raise ValueError(
                f'Invalid total charge {total_charges} for balancing ions! '
                'Total charge must be an integer.'
            )

        if total_charges > 0:
            ion_list = [anion for i in range(int(total_charges))]
        else:
            ion_list = [cation for i in range(int(-total_charges))]

        new_ion_chain = self._create_ion_chain(solvents, ion_list)
        self.model.add(new_ion_chain)
        if self._topo_loader is not None:
            self._topo_loader.generate(new_ion_chain)

        return new_ion_chain
        
    def _create_ion_chain(self, solvents, ion_list):
        
        water_res = [res for chain in solvents for res in chain]
        chosen_waters = choices(water_res, k=len(ion_list))
        rtf = ResidueTopologySet('water_ions')
        new_ion_chain = Ion('IA')
        ion_names = ', '.join(set(ion_list))
        new_ion_chain.pdbx_description = f"balancing ions ({ion_names})"
        for i, (chosen_water, ion_name) in enumerate(zip(chosen_waters, ion_list), start=1):
            if 'OH2' in chosen_water:
                oxy_coord = chosen_water['OH2'].coord
            elif 'O' in chosen_water:
                oxy_coord = chosen_water['O'].coord
            else:
                raise KeyError(f'No oxygen atom present in water {chosen_water}')
            if ion_name not in rtf.res_defs:
                raise ValueError(
                    f'Ion {ion_name} not exist in water_ions.rtf. Ion names must be '
                    f'in {CHARMM_PDB_ION_NAMES.keys()}'
                )
            ion_res = rtf[ion_name].create_residue(resseq=i)
            ion_res.atoms[0].coord = oxy_coord
            new_ion_chain.add(ion_res)
            if (water_chain:=chosen_water.parent) is not None:
                water_chain.detach_child(chosen_water.id)

        return new_ion_chain


