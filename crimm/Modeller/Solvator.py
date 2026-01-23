"""This module contains the Solvator class, which solvates a Structure, Model,
or Chain level entity with water molecules.
"""
import os
import warnings
import numpy as np
import math
from typing import Optional, Tuple
from dataclasses import dataclass
import random
from random import choices
from scipy.spatial import KDTree
from crimm import Data
from crimm.Modeller.CoordManipulator import CoordManipulator
from crimm.StructEntities.Atom import Atom
from crimm.StructEntities.Residue import Residue
from crimm.StructEntities.Model import Model
from crimm.StructEntities.Chain import Solvent, Ion
from crimm.Data.components_dict import CHARMM_PDB_ION_NAMES
from crimm.Modeller.TopoLoader import TopologyGenerator


WATER_COORD_PATH = os.path.join(os.path.dirname(Data.__file__), 'water_coords.npy')
BOXWIDTH=18.662 # water unit cube width

# Physical constants for ion calculations
WATER_MOLARITY = 55.5  # Molar concentration of pure water at 298K (often approximated as 56)
AVOGADRO = 6.02214076e23  # mol^-1
NM3_TO_L = 1e-24  # nm³ to liters conversion
A3_TO_NM3 = 1e-3  # Å³ to nm³ conversion
WATERS_PER_NM3 = 33.3  # Approximate number of water molecules per nm³ at 298K

# Crystal type definition
@dataclass
class CrystalType:
    """CHARMM crystal type definition.

    Attributes
    ----------
    name : str
        Display name for the crystal type
    charmm_name : str
        CHARMM 4-character code (CUBI, OCTA, RHDO, etc.)
    dof : int
        Degrees of freedom for lattice optimization
    volume_factor : float
        Volume relative to cubic box with same max dimension (for isotropic types)
    angles : Tuple[float, float, float]
        Crystal angles (alpha, beta, gamma) in degrees
    supports_solvation : bool
        Whether water box generation is supported for this type
    """
    name: str
    charmm_name: str
    dof: int
    volume_factor: float
    angles: Tuple[float, float, float]
    supports_solvation: bool

# All CHARMM crystal types (from ~/software/charmm/doc/crystl.info)
CRYSTAL_TYPES = {
    'cube': CrystalType(
        name='Cubic',
        charmm_name='CUBI',
        dof=1,
        volume_factor=1.0,
        angles=(90.0, 90.0, 90.0),
        supports_solvation=True
    ),
    'octa': CrystalType(
        name='Truncated Octahedron',
        charmm_name='OCTA',
        dof=1,
        volume_factor=0.77,  # (4*sqrt(3)/9) ≈ 0.77
        angles=(109.4712206344907, 109.4712206344907, 109.4712206344907),
        supports_solvation=True
    ),
    'rhdo': CrystalType(
        name='Rhombic Dodecahedron',
        charmm_name='RHDO',
        dof=1,
        volume_factor=0.707,  # sqrt(0.5) ≈ 0.707
        angles=(60.0, 90.0, 60.0),
        supports_solvation=True
    ),
    'ortho': CrystalType(
        name='Orthorhombic',
        charmm_name='ORTH',
        dof=3,
        volume_factor=1.0,  # Variable, depends on a,b,c
        angles=(90.0, 90.0, 90.0),
        supports_solvation=True
    ),
    'tetra': CrystalType(
        name='Tetragonal',
        charmm_name='TETR',
        dof=2,
        volume_factor=1.0,  # Variable, depends on a,c
        angles=(90.0, 90.0, 90.0),
        supports_solvation=True
    ),
    'hexa': CrystalType(
        name='Hexagonal',
        charmm_name='HEXA',
        dof=2,
        volume_factor=0.866,  # sqrt(0.75) ≈ 0.866
        angles=(90.0, 90.0, 120.0),
        supports_solvation=True
    ),
    'mono': CrystalType(
        name='Monoclinic',
        charmm_name='MONO',
        dof=4,
        volume_factor=1.0,
        angles=(90.0, 90.0, 90.0),  # beta varies
        supports_solvation=False
    ),
    'tric': CrystalType(
        name='Triclinic',
        charmm_name='TRIC',
        dof=6,
        volume_factor=1.0,
        angles=(90.0, 90.0, 90.0),  # all vary
        supports_solvation=False
    ),
    'rhomb': CrystalType(
        name='Rhombohedral',
        charmm_name='RHOM',
        dof=2,
        volume_factor=1.0,  # Variable
        angles=(60.0, 60.0, 60.0),  # all equal, <120
        supports_solvation=False
    ),
}

# Volume factors for different box geometries (relative to cubic)
# Kept for backward compatibility
BOX_VOLUME_FACTORS = {k: v.volume_factor for k, v in CRYSTAL_TYPES.items()}

# Ion valence table - only ions defined in crimm/Data/toppar/water_ions.str
ION_VALENCES = {
    # Alkali metals (+1)
    'LIT': +1,   # Lithium
    'SOD': +1,   # Sodium
    'POT': +1,   # Potassium
    'RUB': +1,   # Rubidium
    'CES': +1,   # Cesium
    # Alkaline earth (+2)
    'MG': +2,    # Magnesium
    'CAL': +2,   # Calcium
    'BAR': +2,   # Barium
    # Transition metals (defined in toppar)
    'ZN2': +2,   # Zinc
    'CD2': +2,   # Cadmium
    # Anions
    'CLA': -1,   # Chloride
    'OH': -1,    # Hydroxide
}

# Topology definitions for water and ions
_topo = TopologyGenerator()
_topo_def, _params = _topo.load_residue_definitions('Solvent')

TIP3_def = _topo_def['TIP3']
# Create new Atom instances without coordinates
OH2 = TIP3_def['OH2'].create_new_atom()
H1 = TIP3_def['H1'].create_new_atom()
H2 = TIP3_def['H2'].create_new_atom()

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

    >>> model = fetch_rcsb('5igv')
    >>> solvator = Solvator(model)
    >>> water_chains = solvator.solvate(
            cutoff=8.0, solvcut=2.1, remove_existing_water=True
        )
    >>> solvent_chains = [
        chain for chain in model if chain.chain_type == 'Solvent'
    ]

    Note that water chains are named W[A-Z]. All waters are placed in a single chain
    (PSF/CRD extended format supports residue numbers up to 10 digits).
    >>> water_chains
    [<Solvent id=WA Residues=12485>]
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
        self.box_dims = None  # For orthorhombic: (a, b, c)
        self.water_box_coords = None
        # unit of pre-equilibrated cube of water molecules (18.662 A each side)
        self.water_unit_coords = np.load(WATER_COORD_PATH)
        self._topo_loader = getattr(self.model, 'topology_loader', None)
        self.box_type = None
        self.orient_method = None
        self.crystal_type = None  # CrystalType object
        
    def get_model(self):
        return self.model

    def suggest_optimal_crystal(
            self,
            cutoff: float = 9.0,
            candidates: list = None
        ) -> dict:
        """Suggest optimal crystal type based on molecule geometry.

        Analyzes the molecule's bounding box dimensions after PCA orientation
        and calculates the solvent volume for each candidate crystal type.
        Returns the type that minimizes total solvent volume.

        The function simulates the orientation that would be applied during
        solvation to provide accurate volume estimates.

        Parameters
        ----------
        cutoff : float, optional
            Minimum distance from solute to box edge in Angstroms (default: 9.0)
        candidates : list, optional
            List of crystal types to consider.
            Default: ['cube', 'octa', 'rhdo', 'ortho']

        Returns
        -------
        dict
            Dictionary with:
            - 'recommended': str, the optimal crystal type
            - 'reason': str, explanation for the recommendation
            - 'volumes': dict, volume comparison for all candidates (in Å³)
            - 'molecule_shape': str, 'globular', 'elongated', or 'intermediate'
            - 'extents': tuple, molecule dimensions after orientation (sorted: largest first)
            - 'aspect_ratios': dict, shape descriptors

        Examples
        --------
        >>> solvator = Solvator(model)
        >>> result = solvator.suggest_optimal_crystal(cutoff=10.0)
        >>> print(f"Recommended: {result['recommended']}")
        >>> print(f"Reason: {result['reason']}")
        >>> solvator.solvate(cutoff=10.0, box_type=result['recommended'])
        """
        if candidates is None:
            candidates = ['cube', 'octa', 'rhdo', 'ortho']

        # Validate candidates
        for c in candidates:
            if c not in CRYSTAL_TYPES:
                raise ValueError(f"Unknown crystal type: {c}")
            if not CRYSTAL_TYPES[c].supports_solvation:
                raise ValueError(f"Crystal type '{c}' does not support solvation")

        # Extract coordinates from model EXCLUDING water and ion chains
        # This matches solvation behavior where waters are removed before orientation
        coords = self._extract_coords(self.model, non_solvent_only=True)

        # Simulate orient_coords_ortho: PCA with axis sorting by singular values
        # This matches exactly what happens during solvation with ortho orientation
        centered = coords - coords.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Sort principal components by decreasing singular values (largest variance first)
        sort_idx = np.argsort(S)[::-1]
        rotation = Vt[sort_idx].T
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        
        # Apply rotation to get oriented coordinates
        oriented_coords = centered @ rotation
        
        # Get extents after orientation (now sorted: largest on X, smallest on Z)
        extents = np.ptp(oriented_coords, axis=0)
        
        # These are already sorted by the SVD-based orientation
        # extents[0] >= extents[1] >= extents[2] (approximately)
        sorted_extents = extents  # Already sorted by orientation
        
        # Calculate aspect ratios
        aspect_xy = sorted_extents[0] / sorted_extents[1] if sorted_extents[1] > 0 else 1.0
        aspect_xz = sorted_extents[0] / sorted_extents[2] if sorted_extents[2] > 0 else 1.0

        # Classify molecule shape
        if aspect_xy < 1.3 and aspect_xz < 1.5:
            shape = 'globular'
        elif aspect_xz > 2.5:
            shape = 'elongated'
        else:
            shape = 'intermediate'

        # Calculate volumes for each candidate
        volumes = {}
        for crystal_type in candidates:
            if crystal_type == 'ortho':
                # Orthorhombic: use actual extents from orientation
                dims = sorted_extents + 2 * cutoff
                vol = dims[0] * dims[1] * dims[2]
            elif crystal_type == 'tetra':
                # Tetragonal: a=b (max of X,Y), c (Z)
                a = max(sorted_extents[0], sorted_extents[1]) + 2 * cutoff
                c = sorted_extents[2] + 2 * cutoff
                vol = a * a * c
            elif crystal_type == 'hexa':
                # Hexagonal: a=b (max of X,Y), c (Z with smallest extent)
                # Volume = sqrt(3)/2 * a² * c ≈ 0.866 * a² * c
                a = max(sorted_extents[0], sorted_extents[1]) + 2 * cutoff
                c = sorted_extents[2] + 2 * cutoff
                vol = 0.866 * a * a * c
            else:
                # Isotropic types (cube, octa, rhdo) use max extent
                box_dim = sorted_extents.max() + 2 * cutoff
                vol_factor = CRYSTAL_TYPES[crystal_type].volume_factor
                vol = vol_factor * (box_dim ** 3)
            volumes[crystal_type] = vol

        # Find minimum volume
        recommended = min(volumes, key=volumes.get)
        min_vol = volumes[recommended]
        cube_vol = volumes.get('cube', min_vol)

        # Generate reason with savings percentage
        savings = (cube_vol - min_vol) / cube_vol * 100 if cube_vol > 0 else 0

        if recommended == 'ortho':
            if shape == 'elongated':
                reason = (f"Molecule is elongated (aspect ratio {aspect_xz:.1f}:1). "
                         f"Orthorhombic box saves {savings:.0f}% volume vs cubic.")
            else:
                reason = (f"Orthorhombic fits molecule dimensions best, "
                         f"saving {savings:.0f}% volume vs cubic.")
        elif recommended in ('octa', 'rhdo'):
            reason = (f"Molecule is {shape}. {CRYSTAL_TYPES[recommended].name} "
                     f"saves {savings:.0f}% volume vs cubic.")
        else:
            reason = f"{CRYSTAL_TYPES[recommended].name} minimizes solvent volume ({savings:.0f}% savings)."

        return {
            'recommended': recommended,
            'reason': reason,
            'volumes': volumes,
            'molecule_shape': shape,
            'extents': tuple(sorted_extents),
            'aspect_ratios': {'xy': aspect_xy, 'xz': aspect_xz}
        }

    def solvate(
            self, cutoff=9.0, solvcut = 2.10,
            remove_existing_water = True,
            remove_existing_ions = False,
            orient_coords = True,
            box_type = 'cube',
            orient_method = None,
            box_dims = None

        ) -> list:
        """Solvate the entity with a water box.

        The solvated entity will be centered in a water box with dimensions
        based on the chosen crystal type. Coordinates will be oriented using
        CoordManipulator before solvation. The solvcut distance is used to
        remove water molecules that are too close to the solute.

        The model is modified in place - water chains are added directly to the
        model. The returned list contains the added water chains.

        Parameters
        ----------
        cutoff : float, optional
            The distance from the solute to the edge of the box (default: 9.0 Å)
        solvcut : float, optional
            The distance from the solute at which water molecules will be
            removed (default: 2.10 Å)
        remove_existing_water : bool, optional
            If True, remove existing water molecules from the entity (default: True)
        remove_existing_ions : bool, optional
            If True, remove existing ions from the entity (default: False)
        orient_coords : bool, optional
            If True, orient coordinates before solvation (default: True)
        box_type : str, optional
            The shape of the water box. Supported types:
            - 'cube': Cubic box (default)
            - 'octa': Truncated octahedron (~23% volume savings)
            - 'rhdo': Rhombic dodecahedron (~29% volume savings)
            - 'ortho': Orthorhombic (rectangular, optimal for elongated molecules)
            - 'tetra': Tetragonal (a=b, c variable)
            - 'hexa': Hexagonal prism
            Use suggest_optimal_crystal() to get recommendations.
        orient_method : str, optional
            The method to orient the coordinates before solvation:
            - None or 'default': Standard orientation (major axis along X)
            - 'octa': PCA-based orientation for octahedral/rhdo boxes
        box_dims : tuple, optional
            For orthorhombic/tetragonal boxes, specify (a, b, c) dimensions.
            If None, dimensions are calculated from molecule extents + cutoff.

        Returns
        -------
        list
            List of Solvent chains added to the model.

        See Also
        --------
        suggest_optimal_crystal : Recommend optimal crystal type based on geometry
        CRYSTAL_TYPES : Dictionary of all supported crystal types
        """
        # Validate crystal type
        if box_type not in CRYSTAL_TYPES:
            raise ValueError(
                f"Unknown box_type '{box_type}'. "
                f"Supported types: {list(CRYSTAL_TYPES.keys())}"
            )

        self.crystal_type = CRYSTAL_TYPES[box_type]

        if not self.crystal_type.supports_solvation:
            raise ValueError(
                f"Crystal type '{box_type}' does not support water box generation. "
                f"Solvation-compatible types: "
                f"{[k for k, v in CRYSTAL_TYPES.items() if v.supports_solvation]}"
            )

        self.cutoff = cutoff
        self.solvcut = solvcut
        self.box_type = box_type
        self.orient_method = orient_method

        # Set default orientation method based on box type
        # Each crystal type benefits from a specific orientation strategy
        if self.orient_method is None:
            if self.box_type in ("octa", "rhdo", "cube"):
                # PCA minimizes bounding box for isotropic geometries
                self.orient_method = "octa"
            elif self.box_type in ("ortho", "tetra"):
                # Sort axes by extent: largest→X, medium→Y, smallest→Z
                self.orient_method = "ortho"
            elif self.box_type == "hexa":
                # Hexagonal: put flattest dimension on Z (prism height)
                self.orient_method = "hexa"
            else:
                self.orient_method = "default"
        
        # Initialize solvation info storage
        if not hasattr(self.model, '_solvation_info'):
            self.model._solvation_info = {}

        if remove_existing_water:
            self.remove_existing_water(self.model)
            self.model._solvation_info['preserved_waters'] = 0
        else:
            # Convert existing waters to TIP3 format for CHARMM compatibility
            n_preserved_waters = self._convert_existing_waters_to_tip3(self.model)
            self.model._solvation_info['preserved_waters'] = n_preserved_waters

        if remove_existing_ions:
            self.remove_existing_ions(self.model)
            self.model._solvation_info['preserved_ions'] = 0
        else:
            # preserve existing ions (convert names if needed)
            n_preserved_ions = self._convert_existing_ions(self.model)
            self.model._solvation_info['preserved_ions'] = n_preserved_ions

        if len(self.model.chains) == 0:
            raise ValueError('No chains in model to solvate')
        if orient_coords:
            coorman = CoordManipulator()
            coorman.load_entity(self.model)
            apply_parent = self.model.parent is not None

            if self.orient_method == "octa":
                coorman.orient_coords_octa(apply_to_parent=apply_parent)
                warnings.warn("Using PCA orientation for solvation.", UserWarning)
            elif self.orient_method == "ortho":
                coorman.orient_coords_ortho(apply_to_parent=apply_parent)
                warnings.warn("Using orthorhombic orientation for solvation.", UserWarning)
            elif self.orient_method == "hexa":
                coorman.orient_coords_hexa(apply_to_parent=apply_parent)
                warnings.warn("Using hexagonal orientation for solvation.", UserWarning)
            else:
                coorman.orient_coords(apply_to_parent=apply_parent)
                warnings.warn("Using default orientation for solvation.", UserWarning)

            warnings.warn(
                'Orienting coordinates before solvation. This may change the '
                'atom coordinates of the entity in the structure.',
                UserWarning
            )
        self.coords = self._extract_coords(self.model, non_solvent_only=True)
        # Box dimension cannot be smaller than the existing solvent molecule extents
        cur_solvent_dim = self._extract_coords(self.model, non_solvent_only=False)
        min_box_dim = np.ptp(cur_solvent_dim, axis=0)

        # Calculate box dimensions based on crystal type
        extents = np.ptp(self.coords, axis=0)
        
        if self.box_type in ('cube', 'octa', 'rhdo'):
            # Isotropic boxes: use max extent
            self.box_dim = extents.max() + 2 * self.cutoff
            self.box_dims = (self.box_dim, self.box_dim, self.box_dim)
        elif self.box_type == 'ortho':
            # Orthorhombic: use actual extents for each dimension
            if box_dims is not None:
                self.box_dims = tuple(box_dims)
            else:
                self.box_dims = tuple(extents + 2 * self.cutoff)
            self.box_dim = max(self.box_dims)  # For compatibility
        elif self.box_type == 'tetra':
            # Tetragonal: a=b, c can be different
            if box_dims is not None:
                self.box_dims = tuple(box_dims)
            else:
                a = max(extents[0], extents[1]) + 2 * self.cutoff
                c = extents[2] + 2 * self.cutoff
                self.box_dims = (a, a, c)
            self.box_dim = max(self.box_dims)
        elif self.box_type == 'hexa':
            # Hexagonal: a=b, c can be different
            if box_dims is not None:
                self.box_dims = tuple(box_dims)
            else:
                a = max(extents[0], extents[1]) + 2 * self.cutoff
                c = extents[2] + 2 * self.cutoff
                self.box_dims = (a, a, c)
            self.box_dim = max(self.box_dims)
        else:
            # Fallback for any other type
            self.box_dim = extents.max() + 2 * self.cutoff
            self.box_dims = (self.box_dim, self.box_dim, self.box_dim)
        if self.box_dims[0] < min_box_dim[0] or \
           self.box_dims[1] < min_box_dim[1] or \
           self.box_dims[2] < min_box_dim[2]:
            raise ValueError(
                f'Calculated box dimensions {self.box_dims}Å are smaller than existing '
                f'solvent molecule extents {min_box_dim}Å. Increase cutoff or provide '
                'larger box_dims for orthorhombic/tetragonal boxes.'
            )
        return self._solvate_model()


    def _extract_coords(self, entity, non_solvent_only: bool = False) -> np.ndarray:
        """Extracts coordinates from entity. If any altloc atoms are present,
        only the first altloc atoms will be included in the returned array.
        Parameters
        ----------
        entity : Structure, Model, or Chain level entity
            The entity from which to extract coordinates.
        non_solvent_only : bool, optional
            If True and entity is Model level, exclude water and ion chains.
            Default is False.
        Returns
        -------
        np.ndarray
            Array of coordinates.
        """
        # get non-solvent atoms only if specified
        if non_solvent_only and entity.level == 'M':
            atoms = []
            for chain in entity:
                chain_type = getattr(chain, 'chain_type', None)
                # Skip water and ion chains
                if chain_type in ('Solvent', 'Ion'):
                    continue
                for atom in chain.get_atoms(include_alt=False):
                    atoms.append(atom)
        else:
            # get all atoms
            atoms = entity.get_atoms(include_alt=False)
        
        coords = [atom.coord for atom in atoms]
        return np.array(coords)

    def remove_existing_water(self, model: Model) -> Model:
        """Removes existing water molecules from the model."""
        remove_list = []
        for chain in model.chains:
            if chain.chain_type == 'Solvent':
                remove_list.append(chain.id)
            elif chain.chain_type == 'Ion' and chain.source == 'generated':
                # Also remove generated ions (not crystallographic)
                remove_list.append(chain.id)
        for chain_id in remove_list:
            warnings.warn(
                f'Removing existing water chain {chain_id} from model',
                UserWarning
            )
            model.detach_child(chain_id)
    
    def remove_existing_ions(self, model: Model) -> Model:
        """Removes existing ions from the model."""
        remove_list = []
        for chain in model.chains:
            if chain.chain_type == 'Ion':
                remove_list.append(chain.id)
        for chain_id in remove_list:
            warnings.warn(
                f'Removing existing ion chain {chain_id} from model',
                UserWarning
            )
            model.detach_child(chain_id)

    def _build_water_hydrogens(self, oxygen_coord: np.ndarray) -> tuple:
        """Build hydrogen positions for a water molecule from oxygen position.

        Uses TIP3P geometry:
        - O-H bond length: 0.9572 Å
        - H-O-H angle: 104.52°

        Parameters
        ----------
        oxygen_coord : np.ndarray
            Oxygen atom coordinates (x, y, z).

        Returns
        -------
        tuple
            (h1_coord, h2_coord) as numpy arrays.
        """
        # TIP3P geometry
        oh_bond = 0.9572  # Å
        hoh_angle = 104.52 * np.pi / 180  # radians

        # Generate random orientation
        # Create a random unit vector for first O-H bond
        random_vec = np.random.randn(3)
        random_vec /= np.linalg.norm(random_vec)

        # First hydrogen
        h1_coord = oxygen_coord + oh_bond * random_vec

        # Second hydrogen: rotate around a perpendicular axis
        # Find a perpendicular vector
        perp = np.cross(random_vec, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 0.1:
            perp = np.cross(random_vec, np.array([0, 1, 0]))
        perp /= np.linalg.norm(perp)

        # Rotate random_vec by HOH angle around perpendicular axis
        cos_a = np.cos(hoh_angle)
        sin_a = np.sin(hoh_angle)
        # Rodrigues rotation formula
        h2_vec = (random_vec * cos_a +
                  np.cross(perp, random_vec) * sin_a +
                  perp * np.dot(perp, random_vec) * (1 - cos_a))
        h2_coord = oxygen_coord + oh_bond * h2_vec

        return h1_coord, h2_coord
        
    def _convert_existing_waters_to_tip3(self, model: Model) -> int:
        """Convert existing water molecules (HOH/WAT) to CHARMM TIP3 format.

        This:
        - Renames residue: HOH/WAT -> TIP3
        - Renames oxygen: O/OW -> OH2
        - Builds missing hydrogens for crystallographic waters

        Returns
        -------
        int
            Number of crystal waters preserved/converted
        """
        # Standard atom name mappings for water oxygen
        oxygen_names = {'O', 'OW', 'OH2'}

        n_converted = 0
        n_hydrogens_built = 0

        for chain in model.chains:
            if chain.chain_type != 'Solvent':
                continue
            chain.source = 'crystallographic'  # mark as original waters
            for residue in chain.get_residues():
                if residue.resname in ('HOH', 'WAT', 'SOL', 'TIP3'):
                    atoms = list(residue.get_atoms())

                    # Find oxygen atom
                    oxygen_atom = None
                    hydrogen_atoms = []
                    for atom in atoms:
                        if atom.name in oxygen_names or atom.element == 'O':
                            oxygen_atom = atom
                        elif atom.element == 'H':
                            hydrogen_atoms.append(atom)

                    if oxygen_atom is None:
                        # No oxygen found - skip this residue
                        continue

                    # Build missing hydrogens if needed
                    if len(hydrogen_atoms) < 2:
                        h1_coord, h2_coord = self._build_water_hydrogens(oxygen_atom.coord)
                        n_hydrogens_built += (2 - len(hydrogen_atoms))

                        # Get next available serial number
                        max_serial = max(a.serial_number for a in atoms) if atoms else 0

                        if len(hydrogen_atoms) == 0:
                            # Build both hydrogens
                            h1 = H1.copy()
                            h2 = H2.copy()
                            h1.coord = h1_coord
                            h2.coord = h2_coord
                            residue.add(h1)
                            residue.add(h2)
                        elif len(hydrogen_atoms) == 1:
                            h2 = H2.copy()
                            h2.coord = h2_coord
                            residue.add(h2)

                    # Convert residue name to TIP3
                    if residue.resname != 'TIP3':
                        residue.resname = 'TIP3'
                        n_converted += 1

                    # Rename oxygen atom and set topology definition
                    if oxygen_atom.name != 'OH2':
                        oxygen_atom.name = 'OH2'
                        oxygen_atom.fullname = ' OH2'
                    # Ensure oxygen has proper topology definition (charge=-0.834, type=OT)
                    oxygen_atom.topo_definition = OH2.topo_definition

        # Count total preserved waters (converted + already TIP3)
        n_preserved = 0
        for chain in model.chains:
            if chain.chain_type == 'Solvent':
                n_preserved = len(list(chain.get_residues()))
                break

        if n_converted > 0:
            warnings.warn(
                f'Converted {n_converted} existing water molecules to TIP3 format',
                UserWarning
            )
        if n_hydrogens_built > 0:
            warnings.warn(
                f'Built {n_hydrogens_built} missing hydrogens for crystallographic waters',
                UserWarning
            )

        return n_preserved

    def _convert_existing_ions(self, model: Model) -> int:
        """Convert existing ions to CHARMM-compatible names.

        Only converts ions that are defined in crimm/Data/toppar/water_ions.str.
        Maps common PDB ion names to CHARMM residue names.

        Returns
        -------
        int
            Number of crystal ions preserved
        """
        # Map PDB ion names to CHARMM names (only toppar-defined ions)
        ion_name_map = {
            # Alkali metals
            'LI': 'LIT', 'LI+': 'LIT',
            'NA': 'SOD', 'NA+': 'SOD', 'NAI': 'SOD',
            'K': 'POT', 'K+': 'POT',
            'RB': 'RUB', 'RB+': 'RUB',
            'CS': 'CES', 'CS+': 'CES',
            # Alkaline earth
            'MG': 'MG', 'MG2+': 'MG', 'MG2': 'MG',
            'CA': 'CAL', 'CA2+': 'CAL', 'CA2': 'CAL',
            'BA': 'BAR', 'BA2+': 'BAR', 'BA2': 'BAR',
            # Transition metals (defined in toppar)
            'ZN': 'ZN2', 'ZN2+': 'ZN2',
            'CD': 'CD2', 'CD2+': 'CD2',
            # Anions
            'CL': 'CLA', 'CL-': 'CLA',
        }

        n_converted = 0
        n_preserved = 0
        for chain in model.chains:
            if chain.chain_type != 'Ion':
                continue
            for residue in chain.get_residues():
                n_preserved += 1
                orig_name = residue.resname.strip().upper()
                if orig_name in ion_name_map:
                    new_name = ion_name_map[orig_name]
                    if residue.resname != new_name:
                        residue.resname = new_name
                        # Also rename the atom to match residue
                        for atom in residue.get_atoms():
                            atom.name = new_name
                            atom.fullname = f' {new_name:<3s}'
                        n_converted += 1

        if n_converted > 0:
            warnings.warn(
                f'Converted {n_converted} existing ions to CHARMM format',
                UserWarning
            )

        return n_preserved

    def create_water_box_coords(self) -> np.ndarray:
        """
        Creates a water box grid based on the chosen box type.

        Supported box types:
          - 'cube': Cubic grid with side length = box_dim
          - 'octa': Truncated octahedron (bounding cube = box_dim * sqrt(4/3))
          - 'rhdo': Rhombic dodecahedron (bounding cube = box_dim * sqrt(2))
          - 'ortho': Orthorhombic box with dimensions box_dims = (a, b, c)
          - 'tetra': Tetragonal box (a=b, c different)
          - 'hexa': Hexagonal prism

        Returns water molecules as array of shape (N_waters, 3, 3).
        """
        from crimm.Modeller.CrystalSDF import (
            sdf_cube, sdf_truncated_octahedron, sdf_rhombic_dodecahedron,
            sdf_orthorhombic, sdf_hexagonal, get_bounding_cube_size
        )

        # Determine grid dimensions based on crystal type
        if self.box_type == 'cube':
            grid_dims = self.box_dims
        elif self.box_type == 'octa':
            # Bounding cube for truncated octahedron
            grid_length = self.box_dim * math.sqrt(4 / 3)
            grid_dims = (grid_length, grid_length, grid_length)
        elif self.box_type == 'rhdo':
            # Bounding cube for rhombic dodecahedron
            grid_length = self.box_dim * math.sqrt(2)
            grid_dims = (grid_length, grid_length, grid_length)
        elif self.box_type in ('ortho', 'tetra'):
            grid_dims = self.box_dims
        elif self.box_type == 'hexa':
            # Hexagon fits in 2a x 2a square (approximately)
            a, _, c = self.box_dims
            grid_dims = (a * 1.2, a * 1.2, c)
        else:
            grid_dims = self.box_dims

        # Build water grid
        n_units = [int(math.ceil(d / BOXWIDTH)) for d in grid_dims]
        water_coords_expanded = self.water_unit_coords.reshape(-1, 3)

        # Generate grid points
        water_points = []
        for i in range(n_units[0]):
            for j in range(n_units[1]):
                for k in range(n_units[2]):
                    translation = np.array([i * BOXWIDTH, j * BOXWIDTH, k * BOXWIDTH])
                    for atom in water_coords_expanded:
                        water_points.append(atom + translation)

        water_points = np.array(water_points)

        # Recenter grid at origin
        translation_vec = -np.ptp(water_points, axis=0) / 2 - water_points.min(0)
        water_points += translation_vec

        # Reshape into water molecules (each with 3 atoms: O, H1, H2)
        water_box = water_points.reshape(-1, 3, 3)

        # For cubic box, return all waters within boundary (no SDF needed)
        if self.box_type == 'cube':
            # Filter waters within cubic boundary
            half = self.box_dim / 2
            selected_waters = []
            for water in water_box:
                oxygen = water[0]
                if all(abs(oxygen) <= half):
                    selected_waters.append(water)
            return np.array(selected_waters) if selected_waters else np.empty((0, 3, 3))

        # For other types, use SDF to filter waters
        selected_waters = []
        for water in water_box:
            oxygen = water[0]

            if self.box_type == 'octa':
                sdf_value = sdf_truncated_octahedron(oxygen, self.box_dim)
            elif self.box_type == 'rhdo':
                sdf_value = sdf_rhombic_dodecahedron(oxygen, self.box_dim)
            elif self.box_type in ('ortho', 'tetra'):
                sdf_value = sdf_orthorhombic(oxygen, self.box_dims)
            elif self.box_type == 'hexa':
                sdf_value = sdf_hexagonal(oxygen, self.box_dims[0], self.box_dims[2])
            else:
                # Fallback to cubic
                sdf_value = sdf_cube(oxygen, self.box_dim)

            # Select waters at least solvcut distance inside the boundary
            if sdf_value <= -self.solvcut:
                selected_waters.append(water)

        return np.array(selected_waters) if selected_waters else np.empty((0, 3, 3))

    def get_expelled_water_box_coords(self) -> np.ndarray:
        """
        Returns water molecules that are outside the solvcut distance from the solute.
        
        All crystal types (cube, octa, rhdo, ortho, tetra, hexa) are now filtered
        by their respective SDF in create_water_box_coords(), so only solvcut 
        filtering is needed here.
        """
        if self.box_type not in ("cube", "octa", "rhdo", "ortho", "tetra", "hexa"):
            raise ValueError(f"Unsupported box type: {self.box_type}")

        # create_water_box_coords() returns shape (N_waters, 3, 3) for all types
        water_box = self.create_water_box_coords()
        
        if len(water_box) == 0:
            return np.empty((0, 3, 3))
        
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

    def _create_new_water_chain(self, alphabet_index) -> Solvent:
        chain_id = 'W'+self.alphabet[alphabet_index]
        water_chain = Solvent(chain_id)
        water_chain.pdbx_description = 'water'
        water_chain.source = 'generated'
        return water_chain

    def _solvate_model(self):
        self.water_box_coords = self.get_expelled_water_box_coords()
        assert self.water_box_coords.shape[1:] == (3, 3), \
        f'Invalid water box coords shape {self.water_box_coords.shape}'

        # Create single water chain (PSF/CRD extended format supports large residue numbers)
        cur_water_chain = self._create_new_water_chain(0)
        water_chains = [cur_water_chain]
        segid = 'SOLV'

        for i, res_coords in enumerate(self.water_box_coords):
            resseq = i + 1  # Sequential residue numbering starting at 1

            water_res = Residue((' ', resseq, ' '), 'TIP3', segid)

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

        # Store box/crystal information for PSF/CRD title generation
        if not hasattr(self.model, '_solvation_info'):
            self.model._solvation_info = {}
        self.model._solvation_info['box_type'] = self.box_type
        self.model._solvation_info['box_dim'] = self.box_dim
        # Store box_dims for orthorhombic and related types
        if hasattr(self, 'box_dims') and self.box_dims is not None:
            self.model._solvation_info['box_dims'] = self.box_dims
        # Store crystal type info for output writers
        if self.box_type in CRYSTAL_TYPES:
            crystal_info = CRYSTAL_TYPES[self.box_type]
            self.model._solvation_info['charmm_name'] = crystal_info.charmm_name
            self.model._solvation_info['angles'] = crystal_info.angles

        return water_chains

    def _remove_generated_ions(self):
        """Removes any previously generated ion chains from the model."""
        remove_list = []
        for chain in self.model.chains:
            if chain.chain_type == 'Ion' and chain.source == 'generated':
                remove_list.append(chain.id)
        for chain_id in remove_list:
            self.model.detach_child(chain_id)
        if len(remove_list) > 0:
            warnings.warn(
                f'Removed {len(remove_list)} previously generated ion chains',
                UserWarning
            )
            
    def add_balancing_ions(
            self, present_charge = None, cation='SOD', anion='CLA', skip_undefined=True,
            remove_generated_ions=True
        ) -> Ion:
        """DEPRECATED: Use add_ions(concentration=0.0) instead.

        Add balancing ions to the solvated entity to bring total charge to zero.
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
        skip_undefined : bool, optional
            If True, chains without defined topology will be assumed to have zero
            charge. If False, a ValueError will be raised if any chain has undefined
            charge. The default is True.
        remove_generated_ions : bool, optional
            If True, any previously generated balancing ion chains will be removed
            before adding new ions. The default is True.
                
        Returns
        -------
        ion_chain : Chain
            A chain containing the balancing ions.
        """
        warnings.warn(
            "add_balancing_ions() is deprecated. Use add_ions(concentration=0.0) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if remove_generated_ions:
            self._remove_generated_ions()
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
                            f'Chain {chain.id} has no topology definition for atom charge! '
                            'Cannot calculate total charge.'
                        )

                    warnings.warn(
                        f'Chain {chain.id} has no topology definition for atom charge! '
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

        new_ion_chain = Ion('IA')
        new_ion_chain.source = 'generated'
        ion_names = ', '.join(set(ion_list))
        new_ion_chain.pdbx_description = f"balancing ions ({ion_names})"
        for i, (chosen_water, ion_name) in enumerate(zip(chosen_waters, ion_list), start=1):
            if 'OH2' in chosen_water:
                oxy_coord = chosen_water['OH2'].coord
            elif 'O' in chosen_water:
                oxy_coord = chosen_water['O'].coord
            else:
                raise KeyError(f'No oxygen atom present in water {chosen_water}')
            if ion_name not in _topo_def.res_defs:
                raise ValueError(
                    f'Ion {ion_name} not exist in water_ions.rtf. Ion names must be '
                    f'in {CHARMM_PDB_ION_NAMES.keys()}'
                )
            ion_res = _topo_def.res_defs[ion_name].create_residue(resseq=i)
            ion_res.atoms[0].coord = oxy_coord
            new_ion_chain.add(ion_res)
            if (water_chain:=chosen_water.parent) is not None:
                water_chain.detach_child(chosen_water.id)

        return new_ion_chain

    # =========================================================================
    # New Ion Calculation Methods (SPLIT, SLTCAP, Add-Neutralize)
    # =========================================================================
    # TODO: Move this inside of the `get_charge` function in Utils
    def _get_ion_chain_charge(self, chain) -> float:
        """Calculate total charge of an ion chain from known ion valences.

        Since ion chains typically don't have topology, we calculate their
        charge from the ION_VALENCES table based on residue names.

        Parameters
        ----------
        chain : Ion
            An ion chain.

        Returns
        -------
        float
            Total charge of all ions in the chain.
        """
        total = 0.0
        for res in chain.get_residues():
            name = res.resname.strip().upper()
            if name in ION_VALENCES:
                total += ION_VALENCES[name]
            else:
                warnings.warn(
                    f"Unknown ion '{name}' in chain {chain.id}, assuming charge 0. "
                    f"Add to ION_VALENCES if needed.",
                    UserWarning
                )
        return total

    def _calculate_system_charge(self, skip_undefined: bool = True) -> float:
        """Calculate total system charge from all non-solvent chains.

        Includes:
        - Protein/nucleic acid chains (from topology)
        - Existing ion chains (from ION_VALENCES table)

        Parameters
        ----------
        skip_undefined : bool
            If True, assume zero charge for chains without topology.
            If False, raise ValueError.

        Returns
        -------
        float
            Total system charge in electron units.
        """
        total_charge = 0.0
        for chain in self.model:
            if chain.chain_type == 'Solvent':
                continue
            if chain.chain_type == 'Ion':
                # Include existing ion charges (e.g., Zn2+, Mg2+, Ca2+)
                ion_charge = self._get_ion_chain_charge(chain)
                total_charge += ion_charge
                continue
            if chain.total_charge is None:
                if not skip_undefined:
                    raise ValueError(
                        f'Chain {chain.id} has no topology definition for atom charge! '
                        'Cannot calculate total charge.'
                    )
                warnings.warn(
                    f'Chain {chain.id} has no topology definition for atom charge! '
                    'Assuming zero charge.',
                    UserWarning
                )
                continue
            total_charge += chain.total_charge
        return total_charge

    def _count_waters(self) -> int:
        """Count total number of water molecules in solvent chains."""
        n_water = 0
        for chain in self.model:
            if chain.chain_type == 'Solvent':
                n_water += len(list(chain.get_residues()))
        return n_water

    def _calculate_solvent_volume(self) -> float:
        """Calculate solvent volume in Å³ based on box geometry.

        Returns
        -------
        float
            Solvent volume in Å³.
        """
        if self.box_dim is None:
            raise ValueError("Box dimensions not set. Run solvate() first.")

        volume_factor = BOX_VOLUME_FACTORS.get(self.box_type, 1.0)
        return volume_factor * (self.box_dim ** 3)

    def _calculate_n0(self, n_water: int, concentration: float) -> float:
        """Calculate N₀ (number of neutral ion pairs) for target concentration.

        Parameters
        ----------
        n_water : int
            Number of water molecules.
        concentration : float
            Target salt concentration in Molar.

        Returns
        -------
        float
            Number of neutral ion pairs (may be fractional).
        """
        return n_water * concentration / WATER_MOLARITY

    def _select_ion_method(
        self, n_water: int, charge: float, concentration: float
    ) -> tuple:
        """Auto-select best ion calculation method based on N₀/|Q| ratio.

        Parameters
        ----------
        n_water : int
            Number of water molecules.
        charge : float
            System charge in electron units.
        concentration : float
            Target salt concentration in Molar.

        Returns
        -------
        tuple
            (method_name, explanation_string)
        """
        n0 = self._calculate_n0(n_water, concentration)

        if charge == 0:
            return 'split', "Neutral system - all methods equivalent"

        if n0 == 0:
            return 'add_neutralize', "No salt concentration - neutralization only"

        ratio = n0 / abs(charge)

        if ratio >= 2.0:
            return 'split', f"N₀/|Q| = {ratio:.1f} ≥ 2: SPLIT excellent (<1% error)"
        elif ratio >= 1.0:
            return 'split', f"N₀/|Q| = {ratio:.1f} ≥ 1: SPLIT acceptable (~7% error)"
        elif ratio >= 0.5:
            return 'sltcap', f"N₀/|Q| = {ratio:.2f} < 1: Using SLTCAP (SPLIT error ~17%)"
        else:
            return 'sltcap', f"N₀/|Q| = {ratio:.2f} << 1: SLTCAP required (high charge)"

    def _adjust_for_neutralization(
        self, n_cation: float, n_anion: float, charge: float
    ) -> tuple:
        """Adjust rounded ion counts to ensure exact charge neutralization.

        Parameters
        ----------
        n_cation : float
            Calculated cation count (may be fractional).
        n_anion : float
            Calculated anion count (may be fractional).
        charge : float
            System charge in electron units.

        Returns
        -------
        tuple
            (n_cation, n_anion) as integers, adjusted for exact neutralization.
        """
        # Ensure non-negative and round
        n_cation = max(0, round(n_cation))
        n_anion = max(0, round(n_anion))

        # Adjust for exact neutralization
        # Net ion charge = n_cation - n_anion (assumes monovalent ions)
        # For neutralization: n_cation - n_anion = -charge
        net_ion_charge = n_cation - n_anion
        target_ion_charge = -int(round(charge))

        if net_ion_charge != target_ion_charge:
            diff = target_ion_charge - net_ion_charge
            if diff > 0:
                n_cation += 1
            else:
                n_anion += 1

        return int(n_cation), int(n_anion)

    def _ions_split(
        self, charge: float, n_water: int, concentration: float
    ) -> tuple:
        """Calculate ion counts using SPLIT method.

        SPLIT: N± = N₀ ∓ Q/2

        Reference: Machado & Pantano (2020) J. Chem. Theory Comput. 16:1367-1372

        Parameters
        ----------
        charge : float
            System charge in electron units.
        n_water : int
            Number of water molecules.
        concentration : float
            Target salt concentration in Molar.

        Returns
        -------
        tuple
            (n_cation, n_anion) as integers.
        """
        n0 = self._calculate_n0(n_water, concentration)
        n_cation = n0 - charge / 2.0
        n_anion = n0 + charge / 2.0
        return self._adjust_for_neutralization(n_cation, n_anion, charge)

    def _ions_sltcap(
        self, charge: float, n_water: int, concentration: float
    ) -> tuple:
        """Calculate ion counts using SLTCAP method.

        SLTCAP: N± = N₀ × [√(1 + (Q/(2N₀))²) ∓ Q/(2N₀)]

        Reference: Schmit et al. (2018) J. Chem. Theory Comput. 14:1823-1827

        Parameters
        ----------
        charge : float
            System charge in electron units.
        n_water : int
            Number of water molecules.
        concentration : float
            Target salt concentration in Molar.

        Returns
        -------
        tuple
            (n_cation, n_anion) as integers.
        """
        n0 = self._calculate_n0(n_water, concentration)

        if n0 == 0:
            # No salt - just neutralization
            if charge > 0:
                return 0, int(abs(charge))
            elif charge < 0:
                return int(abs(charge)), 0
            else:
                return 0, 0

        ratio = charge / (2 * n0)
        sqrt_term = math.sqrt(1 + ratio ** 2)

        n_cation = n0 * (sqrt_term - ratio)
        n_anion = n0 * (sqrt_term + ratio)

        return self._adjust_for_neutralization(n_cation, n_anion, charge)

    def _ions_add_neutralize(
        self, charge: float, n_water: int, concentration: float
    ) -> tuple:
        """Calculate ion counts using Add-Neutralize (AN) method.

        AN method: Add N₀ pairs, then add |Q| counterions.

        WARNING: This method overestimates effective salt concentration
        for charged systems. Use SPLIT or SLTCAP for better accuracy.

        Parameters
        ----------
        charge : float
            System charge in electron units.
        n_water : int
            Number of water molecules.
        concentration : float
            Target salt concentration in Molar.

        Returns
        -------
        tuple
            (n_cation, n_anion) as integers.
        """
        n0 = self._calculate_n0(n_water, concentration)
        n0_int = round(n0)

        if charge > 0:
            n_cation = n0_int
            n_anion = n0_int + int(abs(charge))
        elif charge < 0:
            n_cation = n0_int + int(abs(charge))
            n_anion = n0_int
        else:
            n_cation = n0_int
            n_anion = n0_int

        return int(n_cation), int(n_anion)

    def _place_ions_monte_carlo(
        self,
        solvents: list,
        ion_list: list,
        min_dist_solute: float = 5.0,
        min_dist_ion: float = 5.0,
        max_attempts: int = 1000
    ) -> list:
        """Place ions using Monte-Carlo with distance constraints.

        Parameters
        ----------
        solvents : list
            List of Solvent chain objects.
        ion_list : list
            List of ion names to place (e.g., ['SOD', 'SOD', 'CLA']).
        min_dist_solute : float
            Minimum distance from solute atoms in Å.
        min_dist_ion : float
            Minimum distance between placed ions in Å.
        max_attempts : int
            Maximum attempts to place each ion before giving up.

        Returns
        -------
        list
            List of (water_residue, ion_name) tuples for successful placements.
        """
        # Get all water residues and their oxygen coordinates
        water_residues = []
        water_coords = []
        for chain in solvents:
            for res in chain.get_residues():
                if 'OH2' in res:
                    water_residues.append(res)
                    water_coords.append(res['OH2'].coord)
                elif 'O' in res:
                    water_residues.append(res)
                    water_coords.append(res['O'].coord)

        if len(water_residues) == 0:
            raise ValueError("No water molecules found for ion placement")

        water_coords = np.array(water_coords)

        # Build KDTree of solute atoms
        solute_coords = []
        for chain in self.model:
            if chain.chain_type not in ('Solvent', 'Ion'):
                for atom in chain.get_atoms():
                    solute_coords.append(atom.coord)

        if len(solute_coords) > 0:
            solute_coords = np.array(solute_coords)
            solute_tree = KDTree(solute_coords)
        else:
            solute_tree = None

        # Track placed ion positions and used water indices
        placed_ion_coords = []
        used_water_indices = set()
        placements = []

        for ion_name in ion_list:
            placed = False
            attempts = 0

            # Shuffle water indices for random selection
            available_indices = [
                i for i in range(len(water_residues))
                if i not in used_water_indices
            ]
            random.shuffle(available_indices)

            for idx in available_indices:
                if attempts >= max_attempts:
                    break
                attempts += 1

                coord = water_coords[idx]

                # Check distance from solute
                if solute_tree is not None:
                    dist_to_solute = solute_tree.query(coord)[0]
                    if dist_to_solute < min_dist_solute:
                        continue

                # Check distance from other placed ions
                if len(placed_ion_coords) > 0:
                    ion_coords_array = np.array(placed_ion_coords)
                    dists = np.linalg.norm(ion_coords_array - coord, axis=1)
                    if np.min(dists) < min_dist_ion:
                        continue

                # Accept this placement
                placements.append((water_residues[idx], ion_name))
                placed_ion_coords.append(coord)
                used_water_indices.add(idx)
                placed = True
                break

            if not placed:
                warnings.warn(
                    f"Could not place ion {ion_name} after {max_attempts} attempts. "
                    f"Consider reducing min_dist_solute or min_dist_ion.",
                    UserWarning
                )

        return placements

    def add_ions(
        self,
        concentration: float = 0.15,
        method: str = 'auto',
        cation: str = 'SOD',
        anion: str = 'CLA',
        min_dist_solute: float = 5.0,
        min_dist_ion: float = 5.0,
        skip_undefined: bool = True,
        remove_generated_ions: bool = True,
    ) -> Optional[Ion]:
        """Add ions to achieve target salt concentration.

        Supports three calculation methods:
        - 'split': SPLIT method (Machado & Pantano, 2020) - good for N₀/|Q| ≥ 1
        - 'sltcap': SLTCAP method (Schmit et al., 2018) - accurate for any system
        - 'add_neutralize': Simple add-neutralize (may overestimate concentration)
        - 'auto': Automatically select best method based on N₀/|Q| ratio

        Parameters
        ----------
        concentration : float
            Target salt concentration in Molar. Default 0.15 M (150 mM).
            Use concentration=0 to only neutralize without adding salt.
        method : str
            Ion calculation method: 'auto', 'split', 'sltcap', or 'add_neutralize'.
        cation : str
            CHARMM ion name for cation. Default 'SOD' (Na+).
            Note: Only monovalent ions (SOD, POT, LIT, etc.) are supported.
        anion : str
            CHARMM ion name for anion. Default 'CLA' (Cl-).
            Note: Only monovalent ions (CLA) are supported.
        min_dist_solute : float
            Minimum distance from solute for ion placement in Å. Default 5.0.
        min_dist_ion : float
            Minimum distance between ions in Å. Default 5.0.
        skip_undefined : bool
            If True, assume zero charge for chains without topology. Default True.
        remove_generated_ions : bool
            If True, any previously generated ion chains will be removed. Default True.

        Returns
        -------
        Ion or None
            Chain containing the added ions, or None if no ions needed.

        References
        ----------
        - Schmit et al. (2018) J. Chem. Theory Comput. 14:1823-1827 (SLTCAP)
        - Machado & Pantano (2020) J. Chem. Theory Comput. 16:1367-1372 (SPLIT)
        """
        # Remove previously generated ions
        if remove_generated_ions:
            self._remove_generated_ions()

        # Verify system is solvated
        solvents = [c for c in self.model if c.chain_type == 'Solvent']
        if len(solvents) == 0:
            raise ValueError(
                "Entity must be solvated before adding ions. Run solvate() first."
            )

        # Calculate system properties
        charge = self._calculate_system_charge(skip_undefined)
        n_water = self._count_waters()

        # Select method if auto
        if method == 'auto':
            method, method_reason = self._select_ion_method(n_water, charge, concentration)
        else:
            method_reason = f"User-selected method: {method.upper()}"

        # Calculate ion counts
        if method == 'split':
            n_cation, n_anion = self._ions_split(charge, n_water, concentration)
        elif method == 'sltcap':
            n_cation, n_anion = self._ions_sltcap(charge, n_water, concentration)
        elif method == 'add_neutralize':
            n_cation, n_anion = self._ions_add_neutralize(charge, n_water, concentration)
            # Warn about AN method limitations
            if charge != 0 and concentration > 0:
                n0 = self._calculate_n0(n_water, concentration)
                c_eff = concentration * math.sqrt(1 + abs(charge) / (n0 + 1e-10))
                warnings.warn(
                    f"Add-Neutralize method may overestimate effective salt concentration.\n"
                    f"For system charge Q={charge:.0f}, effective concentration ≈ {c_eff*1000:.0f} mM "
                    f"vs target {concentration*1000:.0f} mM.\n"
                    f"Consider using method='split' or method='sltcap' for better accuracy.",
                    UserWarning
                )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'auto', 'split', 'sltcap', or 'add_neutralize'.")

        # Build ion list
        ion_list = [cation] * n_cation + [anion] * n_anion

        if len(ion_list) == 0:
            print("No ions needed for this system.")
            return None

        # Calculate N₀ for reporting
        n0 = self._calculate_n0(n_water, concentration)
        ratio = n0 / abs(charge) if charge != 0 else float('inf')

        # Print informative message
        print(f"\nIon calculation using {method.upper()} method:")
        print(f"  {method_reason}")
        print(f"  System charge: {charge:+.0f} e")
        print(f"  Water molecules: {n_water:,}")
        print(f"  Target concentration: {concentration*1000:.0f} mM")
        print(f"  N₀ (neutral pairs): {n0:.1f}")
        if charge != 0:
            print(f"  N₀/|Q| ratio: {ratio:.2f}")
        print(f"\n  Adding ions:")
        print(f"    {cation}: {n_cation}")
        print(f"    {anion}: {n_anion}")

        # Place ions using Monte-Carlo
        placements = self._place_ions_monte_carlo(
            solvents, ion_list, min_dist_solute, min_dist_ion
        )

        if len(placements) == 0:
            warnings.warn("No ions could be placed!", UserWarning)
            return None

        # Create ion chain
        new_ion_chain = Ion('IA')
        new_ion_chain.source = 'generated'
        segid = 'IONS'
        ion_names_str = ', '.join(sorted(set(ion_list)))
        new_ion_chain.pdbx_description = f"ions ({ion_names_str}) at {concentration*1000:.0f} mM"

        for i, (water_res, ion_name) in enumerate(placements, start=1):
            if 'OH2' in water_res:
                oxy_coord = water_res['OH2'].coord
            else:
                oxy_coord = water_res['O'].coord

            ion_res = _topo_def.res_defs[ion_name].create_residue(resseq=i)
            ion_res.atoms[0].coord = oxy_coord
            ion_res.segid = segid
            new_ion_chain.add(ion_res)

            # Remove water
            if (water_chain := water_res.parent) is not None:
                water_chain.detach_child(water_res.id)

        self.model.add(new_ion_chain)
        if self._topo_loader is not None:
            self._topo_loader.generate(new_ion_chain)

        # Report final charge
        ion_charge = n_cation - n_anion
        final_charge = charge + ion_charge
        print(f"\n  Final system charge: {final_charge:+.0f} e")
        if abs(final_charge) > 0.5:
            warnings.warn(
                f"System is not fully neutralized (charge = {final_charge:+.0f} e). "
                f"This may cause issues with PME.",
                UserWarning
            )

        # Store solvation metadata for PSF/CRD title generation
        if not hasattr(self.model, '_solvation_info'):
            self.model._solvation_info = {}
        self.model._solvation_info['concentration'] = concentration
        self.model._solvation_info['n_cation'] = n_cation
        self.model._solvation_info['n_anion'] = n_anion
        self.model._solvation_info['cation'] = cation
        self.model._solvation_info['anion'] = anion
        self.model._solvation_info['method'] = method

        return new_ion_chain


