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
        self.op_mat = None
        self._convex_hull = None

    def load_entity(self, entity):
        """Load a structure entity to find translation and rotation operations to
        orient the structure such that the major axis is along the x-axis, and the
        farthest point from the major axis is along the y-axis."""
        self.entity = entity
        self._atoms, self.coords = self._extract_atoms_and_coords(self.entity)
        self.dist_matrix = squareform(pdist(self.coords))
        # pair of indices of the farthest atoms in the structure
        self.end_i, self.end_j = self._find_farthest_atom_indices()
        self.m_translation, self.m_rotation = None, None

    def _extract_atoms_and_coords(self, entity) -> Tuple[List[Atom], np.array]:
        coords = []
        atoms = []
        for atom in entity.get_atoms(include_alt=True):
            coords.append(atom.coord)
            atoms.append(atom)
        return atoms, np.asarray(coords)

    def _find_farthest_atom_indices(self) -> Tuple[int, int]:
        idx_pair = np.unravel_index(
            np.argmax(self.dist_matrix, axis=None),
            self.dist_matrix.shape
        )
        return idx_pair

    def get_transformation_matrix(self) -> Tuple[np.array, np.array]:
        """Return the 4x4 transformation matrix as numpy arrays."""
        if self.op_mat is None:
            self.op_mat = self._find_transformation_operators()
        return self.op_mat

    def _find_transformation_operators(self) -> None:
        a1, a2 = self.end_i, self.end_j
        # coordinates of the farthest atom pair
        o1, o2 = self.coords[a1], self.coords[a2]
        # Translation operator move the center of structure to the origin
        # the center is defined by the midpoint of the major axis 
        # (the line between farthest atoms)
        translation = -(o1 + o2)/2
        major_axis = o1 - o2
        # Translate the structure to the origin
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
        # Find the sign of the x-coordinate of the third atom
        sign = np.sign(norm(c3-c1)-norm(c3-c2))
        c3_x = sign*np.sqrt(np.sum(c3**2) - c3_y**2)

        vec_translated = np.vstack([c1, c2, c3])
        vec_rotated = np.array([
            [-norm(c1), 0, 0],
            [norm(c2), 0, 0],
            [c3_x, c3_y, 0]
        ])
        # Estimate the rotation by Kabsch algorithm implemented in scipy
        rot_obj, rssd = R.align_vectors(vec_rotated, vec_translated)
        # Get the rotation operator as a numpy array
        rotation = rot_obj.as_matrix()
        # Test if the rotation operators can be applied to the translated vectors
        assert np.allclose(vec_translated @ rotation.T, vec_rotated)
        # combine the translation and rotation operators
        combined = np.eye(4)
        # since we translate first, the rotation needs to be applied to the translation vector too
        combined[:3, 3] = translation @ rotation.T
        combined[:3, :3] = rotation

        # find the final translation to recenter the structure
        temp_coords = temp_coords @ rotation.T
        recenter = -(temp_coords.max(0) + temp_coords.min(0))/2
        combined[:3, 3] += recenter

        return combined

    @property
    def coord_center(self):
        """Return the center of the coordinates (N, 3). The center is defined as
        the midpoint of the maximum and minimum coordinates of each dimension. 
        Should be (0, 0, 0) after the transformation by `orient_coords`.
        """
        return (self.coords.max(0) + self.coords.min(0))/2

    @property
    def box_dim(self):
        """Return the dimensions of the bounding box of the coordinates (N, 3).
        The three sides of the box are parallel to the x, y, and z axes.
        """
        return self.coords.ptp(0)

    def apply_coords(self, coords) -> np.array:
        """Apply the transformation to the coordinates (N, 3). Specifically, the 
        transformation is applied as `(coords + translation) @ rotation`, where
        the results are rounded to 4 decimal places."""
        operator = self.get_transformation_matrix()
        homo_coords = np.column_stack((coords, np.ones(len(coords))))
        new_coords = (homo_coords @ operator.T)[:, :3]
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
            # update the coordinates and atoms
            self._atoms, self.coords = self._extract_atoms_and_coords(self.entity)
        else:
            self._apply_to_loaded_entity()
            warnings.warn('No parent structure entity exists!')
