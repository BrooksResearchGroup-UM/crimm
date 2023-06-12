import warnings
from typing import List, Tuple
from Bio.PDB.Atom import Atom
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist, squareform


class CoordManipulator:
    def __init__(self) -> None:
        self.entity = None
        self._atoms = None
        self.coords = None
        self.dist_matrix = None
        self.end_i, self.end_j = None, None
        self.end_ai, self.end_aj = None, None
        self.m_translation, self.m_rotation = None, None

    def load_entity(self, entity):
        """Load a structure entity to find translation and rotation operations to
        orient the structure such that the major axis is along the x-axis, and the
        farthest point from the major axis is along the y-axis."""
        self.entity = entity
        self._atoms, self.coords = self._extract_atoms_and_coords(self.entity)
        self.dist_matrix = squareform(pdist(self.coords))
        # pair of indices of the farthest atoms in the structure
        self.end_i, self.end_j = self._find_farthest_atom_indices()
        # atom handles of the farthest atoms in the structure
        self.end_ai, self.end_aj = self._atoms[self.end_i], self._atoms[self.end_j]
        self.m_translation, self.m_rotation = None, None

    def _extract_atoms_and_coords(self, entity) -> Tuple[List[Atom], np.array]:
        coords = []
        atoms = []
        for atom in entity.get_atoms():
            coords.append(atom.coord)
            atoms.append(atom)
        return atoms, np.asarray(coords)

    def _find_farthest_atom_indices(self) -> Tuple[int, int]:
        idx_pair = np.unravel_index(
            np.argmax(self.dist_matrix, axis=None),
            self.dist_matrix.shape
        )
        return idx_pair

    def get_transformation_matrices(self) -> Tuple[np.array, np.array]:
        """Return the translation and rotation operators as numpy arrays."""
        if self.m_translation is None:
            self._find_transformation_operators()
        return self.m_translation, self.m_rotation

    def _find_transformation_operators(self) -> None:
        a1, a2 = self.end_i, self.end_j
        # coordinates of the farthest atom pair
        c1, c2 = self.coords[a1], self.coords[a2]
        # Translation operator move the center of structure to the origin
        # the center is defined by the midpoint of the major axis 
        # (the line between farthest atoms)
        translation = -(c1 + c2)/2
        major_axis = c1 - c2
        temp_coords = self.coords + translation
        # coordinates of the farthest atom pair after translation
        c1, c2 = temp_coords[a1], temp_coords[a2]
        # Find the third atom that is the farthest from the major axis
        dist_yz = np.abs(
            norm(np.cross(major_axis, temp_coords), axis = 1)/norm(major_axis)
        )
        # coordinates of the third atom after translation but before rotation
        a3 = dist_yz.argmax()
        c3 = temp_coords[a3]
        # coordinates of the third atom after rotation
        c3_y = dist_yz.max()
        c3_x = np.sqrt(norm(c3)**2 - c3_y**2)

        vec_translated = np.vstack([c1, c2, c3])
        vec_rotated = np.array([
            [-norm(c1), 0, 0],
            [norm(c2), 0, 0],
            [c3_x, c3_y, 0]
        ])
        # Estimate the rotation by Kabsch algorithm implemented in scipy
        rot_obj, rssd = R.align_vectors(vec_rotated, vec_translated)
        # Get the rotation operator as a numpy array
        rotation = rot_obj.as_matrix().T
        # Test if the translation and rotation operators can
        # transform a2 to the final coord within error
        test_vecs = np.array([
            self.coords[a1], self.coords[a2], self.coords[a3]
        ])
        assert np.allclose(
            (test_vecs + translation) @ rotation,
            vec_rotated
        )
        self.m_translation, self.m_rotation = translation, rotation

    def apply_coords(self, coords) -> np.array:
        """Apply the transformation to the coordinates (N, 3). Specifically, the 
        transformation is applied as `(coords + translation) @ rotation`, where
        the results are rounded to 4 decimal places."""
        translation, rotation = self.get_transformation_matrices()
        new_coords = (coords + translation) @ rotation
        # round the coordinates to 4 decimal places
        new_coords = np.around(new_coords, decimals=4, out=None)
        return new_coords

    def apply_entity(self, other_entity) -> None:
        """Apply the same transfermation to another structure entity."""
        atoms, coords = self._extract_atoms_and_coords(other_entity)
        new_coords = self.apply_coords(coords)
        for i, atom in enumerate(atoms):
            atom.coord = new_coords[i]

    def _apply_to_loaded_entity(self) -> None:
        self.coords = self.apply_coords(self.coords)
        for i, atom in enumerate(self._atoms):
            atom.coord = self.coords[i]

    def orient_coords(self, apply_to_parent = False) -> None:
        """Apply translation and rotation operations to orient the structure 
        such that the major axis of the structure is placed  along the x-axis, 
        and the farthest point from the major axis is along the y-axis. 
        If apply_to_parent, the same transformation will also be applied to the 
        parent structure entity.
        """
        if self.entity is None:
            raise ValueError('No structure entity loaded!')
        self._find_transformation_operators()
        if not apply_to_parent:
            self._apply_to_loaded_entity()
        elif self.entity.parent is not None:
            self.apply_entity(self.entity.parent)
        else:
            self._apply_to_loaded_entity()
            warnings.warn('No parent structure entity exists!')


