"""This module defines the TopoEntity class and its subclasses Bond, Angle, Dihedral, and Improper."""
from collections import namedtuple
import numpy as np
from Bio.PDB.Selection import unfold_entities

class TopoEntity:
    """A TopoEntity is a base class for Topology entities such as Bonds, Angles, 
    Dihedrals, and Impropers."""
    RED = '\033[91m'
    ENDC = '\033[0m'

    def _create_full_id(self):
        """Create unique id by comparing the end atoms to decide if the sequence need to be flipped"""
        atom_ids = tuple(a.get_full_id() for a in self)
        if atom_ids[0] > atom_ids[-1]:
            atom_ids = tuple(reversed(atom_ids))
        return atom_ids

    def __getnewargs__(self):
        "Support for pickle protocol 2: http://docs.python.org/2/library/pickle.html#pickling-and-unpickling-normal-class-instances"
        return *self, self.param

    def __getstate__(self):
        """
        Additional support for pickle since parent class implements its own __getstate__
        so pickle does not store or restore the type and order, python 2 problem only
        https://www.python.org/dev/peps/pep-0307/#case-3-pickling-new-style-class-instances-using-protocol-2
        """
        return self.__dict__

    def __hash__(self) -> int:
        return hash(self.full_id)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.full_id == other.full_id
 
    def __deepcopy__(self, memo):
        return type(self)(*self, **self.__dict__)

    def get_atom_types(self):
        """Return the atom types of the atoms in this entity"""
        return tuple(a.topo_definition.atom_type for a in self)

    def color_missing(self, atom):
        """Return a string representation of an atom with missing coordinates colored red"""
        if atom.coord is None:
            return f"{self.RED}{atom.name:>4s}{self.ENDC}"
        else:
            return f"{atom.name:>4s}"

    @property
    def parent(self):
        """Return the parent Residue objects in a list"""
        return tuple(unfold_entities(list(self), 'R'))

BondTuple = namedtuple('Bond', ['atom1', 'atom2'])
class Bond(TopoEntity, BondTuple):
    """A Bond object represents a bond between two Atoms within a Topology.

    This class extends tuple, and may be interpreted as a 2-element tuple of Atom objects.
    It also has fields that can optionally be used to describe the bond order and type of bond."""

    bond_order_dict = {'single':1, 'double':2, 'triple':3, 'aromatic':2}
    def __new__(cls, atom1, atom2, bond_type=None, param=None):
        """Create a new Bond. """
        bond = super(Bond, cls).__new__(cls, atom1, atom2)
        bond.type = bond_type
        bond.order = cls.bond_order_dict.get(type)
        bond.param = param
        bond.full_id = bond._create_full_id()
        bond.atom_types = bond.get_atom_types()
        return bond

    def __getnewargs__(self):
        "Support for pickle protocol 2: http://docs.python.org/2/library/pickle.html#pickling-and-unpickling-normal-class-instances"
        return *self, self.type, self.param
    
    def __repr__(self):
        a, b = (self.color_missing(atom) for atom in self)
        s = f"<Bond({a}, {b})"
        if self.type is not None:
            s += f" type={self.type}"
        if self.order is not None:
            s += f" order={self.order:d}"
        if self.length is not None:
            s += f" length={self.length:.2f}"
        s += ">"
        return s

    def __deepcopy__(self, memo):
        return Bond(*self, self.type, self.param)
    
    @property
    def length(self):
        """return the current bond length"""
        if self[0].coord is None or self[1].coord is None:
            return None
        return (((self[0].coord - self[1].coord)**2).sum())**0.5
    
    @property
    def kb(self):
        """return the bond force constant"""
        if self.param is None:
            return None
        return self.param.kb
    
    @property
    def b0(self):
        """return the bond equilibrium length"""
        if self.param is None:
            return None
        return self.param.b0

AngleTuple = namedtuple('Angle', ['atom1', 'atom2', 'atom3'])
class Angle(TopoEntity, AngleTuple):
    """An Angle object represents an angle between three Atoms within a Topology.

    This class extends tuple, and may be interpreted as a 2-element tuple of Bond objects.
    """
    
    def __new__(cls, atom1, atom2, atom3, param=None):
        """Create a new Entity. """
        angle = super(Angle, cls).__new__(cls, atom1, atom2, atom3)
        angle.param = param
        angle.full_id = angle._create_full_id()
        angle.atom_types = angle.get_atom_types()
        return angle
    
    def __repr__(self):
        a, b, c = (self.color_missing(atom) for atom in self)
        s = f"<Angle({a:>4s}, {b:>4s}, {c:>4s})"
        if self.angle is not None:
            s += f" angle={self.angle:.2f}"
        s += ">"
        return s
    
    @property
    def angle(self):
        """return the current angle in degrees"""
        a, b, c = (atom.coord for atom in self)
        for coord in (a, b, c):
            if coord is None:
                return None
        ba = a - b
        bc = c - b
        theta = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
        return np.rad2deg(theta)
    
    @property
    def ktheta(self):
        """return the angle force constant"""
        if self.param is None:
            return None
        return self.param.ktheta
    
    @property
    def theta0(self):
        """return the equilibrium angle"""
        if self.param is None:
            return None
        return self.param.theta0
    
DiheTuple = namedtuple('Dihedral', ['i', 'j', 'k', 'l'])
class Dihedral(TopoEntity, DiheTuple):
    """A Dihedral object represents a dihedral angle between four Atoms within a Topology.

    This class extends tuple, and may be interpreted as a 4-element tuple (i, j, k, l) of 
    Atom objects.
    """

    def __new__(cls, atom_i, atom_j, atom_k, atom_l, param=None):
        """Create a new Entity. """
        dihedral = super(Dihedral, cls).__new__(cls, atom_i, atom_j, atom_k, atom_l)
        dihedral.param = param
        dihedral.full_id = dihedral._create_full_id()
        dihedral.atom_types = dihedral.get_atom_types()
        return dihedral

    def __repr__(self):
        a, b, c, d = (self.color_missing(atom) for atom in self)
        s = f"<Dihedral({a:>4s}, {b:>4s}, {c:>4s}, {d:>4s})"
        if self.angle is not None:
            s += f" angle={self.angle:.2f}"
        s += ">"
        return s

    ## TODO: implement dihedral angle calculation
    @property
    def angle(self):
        """return the current dihedral in degrees"""
        a, b, c, d = (atom.coord for atom in self)
        for coord in (a, b, c, d):
            if coord is None:
                return None
        # ba = a - b
        # dc = d - c
        
        return 0.000


ImprTuple = namedtuple('Improper', ['i', 'j', 'k', 'l'])
class Improper(TopoEntity, ImprTuple):
    """A Dihedral object represents a dihedral angle between four Atoms within a Topology.

    This class extends tuple, and may be interpreted as a 4-element tuple (i, j, k, l) of 
    Atom objects, where i is the center atom.
    """

    def __new__(cls, atom_i, atom_j, atom_k, atom_l, param=None):
        """Create a new Entity. """
        improper = super(Improper, cls).__new__(cls, atom_i, atom_j, atom_k, atom_l)
        improper.param = param
        improper.full_id = improper._create_full_id()
        improper.atom_types = improper.get_atom_types()
        return improper
    
    def __repr__(self):
        a, b, c, d = (self.color_missing(atom) for atom in self)
        s = f"<Improper({a:>4s}, {b:>4s}, {c:>4s}, {d:>4s})"
        if self.angle is not None:
            s += f" angle={self.angle:.2f})"
        s += ">"
        return s
    
    @property
    def angle(self):
        """return the current improper in degrees"""
        a, b, c, d = (atom.coord for atom in self)
        for coord in (a, b, c, d):
            if coord is None:
                return None
        return 0.000

CmapTuple = namedtuple('CMap', ['dihe1', 'dihe2'])
class CMap(TopoEntity, CmapTuple):

    def __new__(cls, dihe1, dihe2, param=None):
        """Create a new Entity. """
        cmap = super(CMap, cls).__new__(cls, dihe1, dihe2)
        cmap.param = param
        cmap.full_id = sorted((dihe1.full_id, dihe2.full_id))
        return cmap
    
    def __repr__(self):
        dihe1, dihe2 = self
        s = f"<Cross-Term({repr(dihe1)}, {repr(dihe2)})"
        if self.angle is not None:
            s += f" angle={self.angle:.2f})"
        s += ">"
        return s
    
    def get_atom_types(self):
        """Return the atom types of the atoms in this entity"""
        dihe1, dihe2 = self
        atom_types1 = dihe1.get_atom_types()
        atom_types2 = dihe2.get_atom_types()
        return tuple(sorted((atom_types1, atom_types2)))
    
    @property
    def angle(self):
        """return the current improper in degrees"""
        a, b, c, d = (atom.coord for atom in self)
        for coord in (a, b, c, d):
            if coord is None:
                return None
        return 0.000