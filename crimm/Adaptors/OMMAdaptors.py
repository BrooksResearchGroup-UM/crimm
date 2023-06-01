import openmm.unit as unit
from openmm.vec3 import Vec3
from openmm.app import element
from openmm.app.topology import Topology as OMMTop
from openmm.app.internal.unitcell import computePeriodicBoxVectors

from crimm.StructEntities.Model import Model
from crimm.StructEntities.Structure import Structure
    
class Topology(OMMTop):
    """Model/Topology class derived from Biopython Model and made compatible with
    OpenMM Topology."""
    def __init__(self, model: Model):
        self._bonds = []
        self._periodicBoxVectors = None
        self.model = model
        self.model.topology = self
        self._assign_periodic_box_vecs()
        self._retype_elements()
        self.createStandardBonds()
        self.createDisulfideBonds(construct_omm_positions(self.model))
    
    def __repr__(self):
        return self.model.__repr__()

    # def _repr_html_(self):
    #     return self.model._repr_html_()

    def get_residues(self):
        """Get a list of residues from all Chains in the Model/Topology"""
        residues = []
        for chain in self.model:
            residues.extend(chain.child_list)
        return residues
    
    @property
    def _chains(self):
        """List of Chains in the Model/Topology
        (OpenMM compatible private attribute)"""
        return self.model.child_list
    
    @property
    def _numAtoms(self):
        """Return the number of atoms in the Topology.
        (OpenMM compatible API)"""
        return len(list(self.model.get_atoms()))

    @property
    def _numResidues(self):
        """Return the number of residues in the Topology.
        (OpenMM compatible API)"""
        return len(list(self.model.get_residues()))

    def getNumChains(self):
        """Return the number of chains in the Topology.
        (OpenMM compatible API)"""
        return len(self.model.child_list)

    def getNumBonds(self):
        """Return the number of bonds in the Topology.
        (OpenMM compatible API)"""
        return len(self._bonds)

    def chains(self):
        """Iterate over all Chains in the Topology.
        (OpenMM compatible API)"""
        return self.model

    def residues(self):
        """Iterate over all Residues in the Topology.
        (OpenMM compatible API)"""
        for chain in self.model:
            for residue in chain:
                yield residue

    def atoms(self):
        """Iterate over all Atoms in the Topology.
        (OpenMM compatible API)"""
        for chain in self.model:
            for residue in chain:
                for atom in residue:
                    yield atom

    def addChain(self, chain):
        """Add a Chain object into the topology"""
        self.model.add(chain)

    def addResidue(self, id=None):
        """OpenMM residue adding methods are disallowed. NotImplementedError will be raised."""
        raise NotImplementedError(
            'OpenMM residue adding methods are disallowed!'
        )
    
    def addAtom(self, id=None):
        """OpenMM atom adding methods are disallowed. NotImplementedError will be raised."""
        raise NotImplementedError(
            'OpenMM atom adding methods are disallowed!'
        )

    def _assign_periodic_box_vecs(self):
        if self.model.parent is None or self.model.parent.cell_info is None:
            return

        cell = self.model.parent.cell_info
        periodic_box_vectors = computePeriodicBoxVectors(
            cell["length_a"],
            cell["length_b"],
            cell["length_c"],
            cell["angle_alpha"],
            cell["angle_beta"],
            cell["angle_gamma"]
        )
        self.setPeriodicBoxVectors(periodic_box_vectors)

    def _retype_elements(self):
        """Convert all the type of element of Atoms from str to OpenMM elements"""
        for atom in self.atoms():
            if isinstance(atom.element, element.Element):
                continue
            atom.element = element.get_by_symbol(atom.element)

def construct_omm_positions(entity):
    if isinstance(entity, Structure):
        # take the first model if a structure is supplied
        entity = entity.child_list[0]

    positions = []
    for atom in entity.get_atoms():
        positions.append(Vec3(*(atom.coord/10)))
    return unit.Quantity(value=positions, unit=unit.nanometers)