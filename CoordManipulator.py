from typing import List, Tuple
from Bio.PDB.Atom import Atom
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist


class CoordManipulator:
    def __init__(self, entity) -> None:
        self.entity = entity
        self._atoms, self.coords = self._extract_atoms_and_coords(entity)

    def _extract_atoms_and_coords(self, entity) -> Tuple[List[Atom], np.array]:
        coords = []
        atoms = []
        for atom in entity.get_atoms():
            coords.append(atom.coord)
            atoms.append(atom)
        return atoms, np.asarray(coords)

    def get_dist_matrix(self) -> np.array:
        if not hasattr(self, "_dist_matrix"):
            self._dist_matrix = pdist(self.coords)
        return self._dist_matrix

    def get_farthest_atom_pair(self) -> Tuple[Atom, Atom]:
        a1, a2 = self.get_farthest_atom_indices()
        # return the atom objects themselves, not the indices
        return self._atoms[a1], self._atoms[a2]

    def get_farthest_atom_indices(self) -> Tuple[int, int]:
        if not hasattr(self, "_a1"):
            self._a1, self._a2 = self._find_farthest_atom_indices()
        # return the atom indices in the dist_mat
        return self._a1, self._a2

    def _find_farthest_atom_indices(self) -> Tuple[int, int]:
        dists = self.get_dist_matrix()
        idx_pair = np.unravel_index(
            np.argmax(dists, axis=None),
            dists.shape
        )
        return idx_pair

    def get_transformation_matrices(self) -> Tuple[np.array, np.array]:
        if not hasattr(self, "m_translation"):
            self._find_transformation_operators()
        return self.m_translation, self.m_rotation

    def _find_transformation_operators(self) -> None:
        dists = self.get_dist_matrix()
        a1, a2 = self.get_farthest_atom_indices()
        # Translation operator is just by moving a1 to origin (0, 0, 0)
        m_translation = -self.coords[a1]
        # Find the coord of a2 after the translation
        vec_translated = (self.coords[a2] + m_translation).reshape(1,-1)
        # Define the final coord of a2 after the rotation (dist(a1,a2), 0, 0),
        # that is, align a1, a2 on the x-axis
        vec_rotated = np.array([dists[a1, a2],0,0]).reshape(1,-1)
        # Estimate the rotation by Kabsch algorithm implemented in scipy
        rot_obj, rssd = R.align_vectors(vec_translated, vec_rotated)
        # Get the rotation operator as a numpy array
        m_rotation = rot_obj.as_matrix()
        # Test if the translation and rotation operators can
        # transform a2 to the final coord within error
        assert np.allclose(
            (self.coords[a2] + m_translation) @ m_rotation,
            vec_rotated
        )
        self.m_translation, self.m_rotation = m_translation, m_rotation
    
    def transform_coords(self, coords) -> np.array:
        translation, rotation = self.get_transformation_matrices()
        new_coords = (coords + translation) @ rotation
        return new_coords

    def apply_transformation(self) -> None:
        for i, atom in enumerate(self._atoms):
            atom.coord = self.coords[i]

    def orient_coords(self) -> None:
        self._find_transformation_operators()
        self.coords = self.transform_coords(self.coords)
        self.apply_transformation()


