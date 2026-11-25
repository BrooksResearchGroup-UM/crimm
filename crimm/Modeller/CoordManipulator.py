import warnings
from typing import List, Tuple

import numpy as np
from Bio.PDB.Atom import Atom
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as R


class CoordManipulator:
    def __init__(self) -> None:
        self.entity = None
        self.include_alt = None
        self._atoms = None
        self.coords = None
        self.dist_matrix = None
        self.end_i, self.end_j = None, None
        self.op_mat = None
        self._convex_hull = None

    def load_entity(self, entity, include_alt=False) -> None:
        """Load a structure entity to find translation and rotation operations to
        orient the structure such that the major axis is along the x-axis, and the
        farthest point from the major axis is along the y-axis."""
        self.entity = entity
        self.include_alt = include_alt
        self._atoms, self.coords = self._extract_atoms_and_coords(
            self.entity, include_alt=self.include_alt
        )
        self.dist_matrix = squareform(pdist(self.coords))
        # pair of indices of the farthest atoms in the structure
        self.end_i, self.end_j = self._find_farthest_atom_indices()
        self.m_translation, self.m_rotation = None, None
        # a transform found for a previously loaded entity does not apply here
        self.op_mat = None

    def _extract_atoms_and_coords(
        self, entity, include_alt
    ) -> Tuple[List[Atom], np.array]:
        coords = []
        atoms = []
        for atom in entity.get_atoms(include_alt=include_alt):
            coords.append(atom.coord)
            atoms.append(atom)
        return atoms, np.asarray(coords)

    def _find_farthest_atom_indices(self) -> Tuple[int, int]:
        idx_pair = np.unravel_index(
            np.argmax(self.dist_matrix, axis=None), self.dist_matrix.shape
        )
        return idx_pair

    def get_transformation_matrix(self) -> np.array:
        """Return the 4x4 transformation matrix `op_mat`.

        This is the transform of the last orientation applied, or, before any,
        the one `orient_coords` would apply. See `_build_op_mat` for the
        convention.
        """
        if self.op_mat is None:
            self.op_mat = self._find_transformation_operators()
        return self.op_mat

    def find_max_dim(self) -> float:
        """Return the largest dimension of the loaded system, in Å.

        This is the distance between the farthest pair of atoms (the major
        axis), which `load_entity` has already found. Nothing is modified.

        Raises
        ------
        ValueError
            If no entity has been loaded.
        """
        if self.entity is None:
            raise ValueError("No structure entity loaded!")
        return float(self.dist_matrix[self.end_i, self.end_j])

    @staticmethod
    def _build_op_mat(rotation, translation) -> np.array:
        """Return the 4x4 matrix of the transform `x -> rotation @ x + translation`.

        This is the one convention for `op_mat`, used by every orientation
        method: column vectors, so the rotation fills `[:3, :3]` and the
        translation fills `[:3, 3]`. A point `x` maps to `op_mat @ [x, 1]`, and
        an (N, 3) array to `[coords, 1] @ op_mat.T` (see `apply_coords`).
        """
        op_mat = np.eye(4)
        op_mat[:3, :3] = rotation
        op_mat[:3, 3] = translation
        return op_mat

    def _find_transformation_operators(self) -> np.array:
        a1, a2 = self.end_i, self.end_j
        # coordinates of the farthest atom pair
        o1, o2 = self.coords[a1], self.coords[a2]
        # Translation operator move the center of structure to the origin
        # the center is defined by the midpoint of the major axis
        # (the line between farthest atoms)
        translation = -(o1 + o2) / 2
        major_axis = o1 - o2
        # Translate the structure to the origin
        temp_coords = self.coords + translation
        # coordinates of the farthest atom pair after translation
        c1, c2 = temp_coords[a1], temp_coords[a2]
        # Find the third atom that is the farthest from the major axis
        dist_yz = np.abs(
            norm(np.cross(major_axis, temp_coords), axis=1) / norm(major_axis)
        )
        # coordinates of the third atom after translation but before rotation
        a3 = dist_yz.argmax()
        c3 = temp_coords[a3]

        # coordinates of the third atom after rotation
        c3_y = dist_yz.max()
        # Find the sign of the x-coordinate of the third atom
        sign = np.sign(norm(c3 - c1) - norm(c3 - c2))
        c3_x = sign * np.sqrt(np.sum(c3**2) - c3_y**2)

        vec_translated = np.vstack([c1, c2, c3])
        vec_rotated = np.array([[-norm(c1), 0, 0], [norm(c2), 0, 0], [c3_x, c3_y, 0]])
        # Estimate the rotation by Kabsch algorithm implemented in scipy
        rot_obj, rssd = R.align_vectors(vec_rotated, vec_translated)
        # Get the rotation operator as a numpy array
        rotation = rot_obj.as_matrix()
        # Test if the rotation operators can be applied to the translated vectors
        assert np.allclose(vec_translated @ rotation.T, vec_rotated)
        # find the final translation to recenter the structure
        temp_coords = temp_coords @ rotation.T
        recenter = -(temp_coords.max(0) + temp_coords.min(0)) / 2
        # since we translate first, the rotation needs to be applied to the
        # translation vector too
        return self._build_op_mat(rotation, translation @ rotation.T + recenter)

    def _find_pca_operators(self, sort_by_extent) -> np.array:
        """Return the 4x4 matrix that moves the centroid to the origin and the
        principal axes onto x, y and z.

        If sort_by_extent, the axes are ordered by decreasing extent (largest
        on x, smallest on z). The axes always form a proper rotation: the SVD
        gives them an arbitrary sign, so when they span a left-handed frame the
        last one is flipped. Otherwise the transform would be a reflection,
        which inverts every chiral centre.
        """
        center = self.coords.mean(axis=0)
        _, S, Vt = np.linalg.svd(self.coords - center, full_matrices=False)
        # rows of Vt are the principal axes
        axes = Vt[np.argsort(S)[::-1]] if sort_by_extent else Vt
        if np.linalg.det(axes) < 0:
            axes[-1] *= -1
        # new = axes @ (x - center), which also centres on the centroid
        return self._build_op_mat(axes, -axes @ center)

    @property
    def coord_center(self):
        """Return the center of the coordinates (N, 3). The center is defined as
        the midpoint of the maximum and minimum coordinates of each dimension.
        Should be (0, 0, 0) after the transformation by `orient_coords`.
        """
        return (self.coords.max(0) + self.coords.min(0)) / 2

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
        atoms, coords = self._extract_atoms_and_coords(other_entity, self.include_alt)
        new_coords = self.apply_coords(coords)
        for i, atom in enumerate(atoms):
            atom.coord = new_coords[i]

    def _apply_to_loaded_entity(self) -> None:
        self.coords = self.apply_coords(self.coords)
        for i, atom in enumerate(self._atoms):
            atom.coord = self.coords[i]

    def _apply_orientation(self, op_mat, apply_to_parent) -> None:
        """Store op_mat and apply it, once, to the loaded entity or its parent.

        If apply_to_parent and a parent exists, the transform is applied to
        every atom of the parent only. The loaded entity's atoms are the
        parent's atoms (shared objects), so they move with it and must not be
        transformed a second time; the cached atoms and coordinates of the
        loaded entity are then read back from them.
        """
        self.op_mat = op_mat
        if not apply_to_parent:
            self._apply_to_loaded_entity()
        elif self.entity.parent is not None:
            self.apply_entity(self.entity.parent)
            # update the coordinates and atoms
            self._atoms, self.coords = self._extract_atoms_and_coords(
                self.entity, self.include_alt
            )
        else:
            self._apply_to_loaded_entity()
            warnings.warn("No parent structure entity exists!")

    def orient_coords(self, apply_to_parent=False) -> None:
        """Apply translation and rotation operations to orient the structure
        such that the major axis of the structure is placed  along the x-axis,
        and the farthest point from the major axis is along the y-axis.
        If apply_to_parent, the same transformation will also be applied to the
        parent structure entity.
        """
        if self.entity is None:
            raise ValueError("No structure entity loaded!")
        self._apply_orientation(self._find_transformation_operators(), apply_to_parent)

    def orient_coords_octa(self, apply_to_parent=False) -> None:
        """
        Apply an alternative orientation aimed at minimizing the bounding
        truncated octahedron. This implementation uses a PCA-based approach to
        align the principal axes with the coordinate axes, and centres the
        structure on its centroid.

        If apply_to_parent, the transformation is applied to the whole parent
        entity (for a model, every model of its structure) instead of only the
        loaded entity. Each atom is moved exactly once, see `_apply_orientation`.
        """
        if self.entity is None:
            raise ValueError("No structure entity loaded!")
        self._apply_orientation(
            self._find_pca_operators(sort_by_extent=False), apply_to_parent
        )

    def orient_coords_ortho(self, apply_to_parent=False) -> None:
        """
        Orient coordinates for orthorhombic box - align principal axes and sort by extent.

        Uses PCA to find principal axes, then sorts them so:
        - Largest extent → X axis (box dimension 'a')
        - Medium extent → Y axis (box dimension 'b')
        - Smallest extent → Z axis (box dimension 'c')

        This orientation minimizes box volume for orthorhombic geometry by ensuring
        that box dimensions match the molecule's extent ordering.

        Note: Same apply_to_parent behavior as orient_coords_octa.
        """
        if self.entity is None:
            raise ValueError("No structure entity loaded!")
        self._apply_orientation(
            self._find_pca_operators(sort_by_extent=True), apply_to_parent
        )

    def orient_coords_hexa(self, apply_to_parent=False) -> None:
        """
        Orient coordinates for hexagonal box - align smallest extent with Z.

        The hexagonal prism has its base in the XY plane (γ=120°).
        For optimal packing, orient molecule so its flattest dimension
        aligns with Z (the prism height), while the two larger dimensions
        span the hexagonal base.

        Axis ordering after orientation:
        - Two largest extents → XY plane (hexagonal base)
        - Smallest extent → Z axis (prism height)

        Note: Same apply_to_parent behavior as orient_coords_octa.
        """
        if self.entity is None:
            raise ValueError("No structure entity loaded!")
        self._apply_orientation(
            self._find_pca_operators(sort_by_extent=True), apply_to_parent
        )
