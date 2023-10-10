# Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)
# Copyright (C) 2023, Truman Xu, Brooks Lab at the University of Michigan
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""Consumer class that builds a Structure object.
This is used by the MMCIFparser classes.
"""
# SMCRA hierarchy
import warnings

from crimm.StructEntities import Structure
from crimm.StructEntities import Model
from crimm.StructEntities import Chain, PolymerChain, Heterogens
from crimm.StructEntities import Residue, DisorderedResidue, Heterogen
from crimm.StructEntities import Atom, DisorderedAtom

class ChainConstructionWarning(Warning):
    """Define class ChainConstructionWarning."""

class ChainConstructionException(Exception):
    """Define class ChainConstructionException."""

class AtomAltLocException(Exception):
    """Define class AtomAltLocException."""


class StructureBuilder():
    """Deals with constructing the Structure object.
    The StructureBuilder class is used by the PDBParser classes to
    translate a file to a Structure object.
    """

    def __init__(self):
        """Initialize the class."""
        self.line_counter = 0
        self.header = None
        self.resolution = None
        self.structure_method = None
        self.structure = None
        self.model = None
        self.chain = None
        self.segid = None
        self.residue = None
        self.atom = None

    def _is_completely_disordered(self, residue):
        """Return 1 if all atoms in the residue have a non blank altloc (PRIVATE)."""
        atom_list = residue.get_unpacked_list()
        for atom in atom_list:
            altloc = atom.get_altloc()
            if altloc == ' ':
                return 0
        return 1

    # Public methods called by the Parser classes
    def set_resolution(self, resolution):
        """Set strucuture resolution."""
        self.resolution = resolution

    def set_structure_method(self, method):
        """Set strucuture method."""
        self.structure_method = method

    def set_header(self, header):
        """Set header."""
        self.header = header

    def set_line_counter(self, line_counter):
        """Tracks line in the PDB file that is being parsed.

        Arguments:
         - line_counter - int

        """
        self.line_counter = line_counter

    def init_structure(self, structure_id):
        """Initialize a new Structure object with given id.

        Arguments:
         - id - string

        """
        self.structure = Structure(structure_id)

    def init_model(self, model_id, serial_num = None):
        """Create a new Model object with given id.

        Arguments:
         - id - int
         - serial_num - int

        """

        self.model = Model(model_id, serial_num)
        self.structure.add(self.model)

    def init_chain(self, chain_id):
        """Create a new Chain object with given id.

        Arguments:
         - chain_id - string

        """
        if self.model.has_id(chain_id):
            self.chain = self.model[chain_id]
            warnings.warn(
                f"WARNING: Chain {chain_id} is discontinuous at "
                f"line {self.line_counter}.",
                ChainConstructionWarning,
            )
        else:
            self.chain = Chain(chain_id)
            self.model.add(self.chain)
            
    def init_seg(self, segid):
        """Flag a change in segid.

        Arguments:
         - segid - string

        """
        self.segid = segid

    def _process_duplicated_res(self, new_residue, duplicate_residue, chain):
        resname = new_residue.resname
        res_id = new_residue.get_id()
        field, resseq, icode = res_id
        if duplicate_residue.is_disordered() == 2:
            # The residue in the chain is a DisorderedResidue object.
            # So just add the last Residue object.
            if duplicate_residue.disordered_has_id(resname):
                # The residue was already made
                # select the residue by resname
                duplicate_residue.disordered_select(resname)
            else:
                # The new residue does not exist in the DisorderedResidue
                # add the new residue to it
                duplicate_residue.disordered_add(new_residue)
            return duplicate_residue
        
        if resname == duplicate_residue.resname:
            # Not disordered but resname and id already exist
            warnings.warn(
                "WARNING: Residue "
                f"('{field}', {resseq}, '{icode}', '{resname}') "
                "already defined with the same name at "
                f"line {self.line_counter}.",
                ChainConstructionWarning,
            )
            return duplicate_residue
        
        # Make a new DisorderedResidue object and put all
        # the Residue objects with the id (field, resseq, icode) in it.
        # These residues each should be completely disordered,
        # i.e. they have non-blank altlocs for all their atoms.
        # If not, the mmCIF file probably contains ambiguous
        # atom altloc definitions (e.g. PDB ID: 2BWX Residue 249 (CYS|CSO))
        if not self._is_completely_disordered(duplicate_residue):
            # if this exception is ignored, a residue will be missing.
            raise AtomAltLocException(
                "Blank altlocs in duplicate residue %s ('%s', %i, '%s')"
                % (resname, field, resseq, icode)
            )

        chain.detach_child(duplicate_residue.id)
        disordered_residue = DisorderedResidue(duplicate_residue.id)
        chain.add(disordered_residue)
        disordered_residue.disordered_add(duplicate_residue)
        disordered_residue.disordered_add(new_residue)
        return disordered_residue

    def add_residue(self, residue, chain):
        """Add a residue to the current chain, and set it as the current residue
         to be built"""
        if not isinstance(chain, PolymerChain):
            # if current chain is not a polymer, we simply add the residue
            # to the chain
            chain.add(residue)
            return residue

        res_id = residue.get_id()
        if chain.has_id(res_id):
            duplicated_res_id = res_id
        else:
            chain.add(residue)
            return residue
        # The residue id is duplicated
        warnings.warn(
            f"WARNING: Residue {res_id} redefined at line {self.line_counter}.",
            ChainConstructionWarning,
        )
        duplicate_residue = chain[duplicated_res_id]
        return self._process_duplicated_res(residue, duplicate_residue, chain)

    def init_residue(self, resname, field, resseq, icode, author_seq_id=None):
        """Create a new Residue object.

        Arguments:
         - resname - string, e.g. "ASN"
         - field - hetero flag, "W" for waters, "H" for
           hetero residues, otherwise blank.
         - resseq - int, sequence identifier
         - icode - string, insertion code

        """
        if field != " ":
            if field == "H":
                # The hetero field consists of H_ + the residue name (e.g. H_FUC)
                field = "H_" + resname
        res_id = (field, resseq, icode)
        if isinstance(self.chain, Heterogens):
            new_residue = Heterogen(res_id, resname, self.segid)
        else:
            new_residue = Residue(
                res_id, resname, self.segid, author_seq_id
            )
        self.residue = self.add_residue(new_residue, self.chain)

    def _assign_atom_names(self, atom, residue):
        # First check if this atom is already present in the residue.
        # If it is, it might be due to the fact that the two atoms have atom
        # names that differ only in spaces (e.g. "CA.." and ".CA.",
        # where the dots are spaces). If that is so, use all spaces
        # in the atom name of the current atom.
        name = atom.name
        fullname = atom.fullname
        if not residue.has_id(name):
            return name

        duplicate_atom = residue[name]
        # atom name with spaces of duplicate atom
        duplicate_fullname = duplicate_atom.get_fullname()

        if duplicate_fullname != fullname:
            # name of current atom now includes spaces
            name = fullname
            warnings.warn(
                f"Atom names {duplicate_fullname} and {fullname} differ "
                f"only in spaces at line {self.line_counter}",
                ChainConstructionWarning,
            )
        return name

    def _create_disordered_atom(self, atom, atom_name, residue):
        disordered_atom = DisorderedAtom(atom_name)
        residue.add(disordered_atom)
        # Add the real atom to the disordered atom, and the
        # disordered atom to the residue
        disordered_atom.disordered_add(atom)
        residue.flag_disordered()

    def _add_to_disordered_atom(self, atom, duplicated_atom_name, residue):
        duplicate_atom = residue[duplicated_atom_name]
        if duplicate_atom.is_disordered() == 2:
            # The duplicate is the DisorderedAtom entity not the Atom object itself
            duplicate_atom.disordered_add(atom)
            return

        # The existing duplicate atom is not a DisorderedAtom
        # This is an error in the PDB file:
        # a disordered atom is found with a blank altloc
        # Detach the duplicate atom, and put it in a
        # DisorderedAtom object together with the current
        # atom.
        residue.detach_child(duplicated_atom_name)
        disordered_atom = DisorderedAtom(duplicated_atom_name)
        residue.add(disordered_atom)
        disordered_atom.disordered_add(atom)
        disordered_atom.disordered_add(duplicate_atom)
        residue.flag_disordered()
        warnings.warn(
            "WARNING: disordered atom found with blank altloc before "
            f"line {self.line_counter}",
            ChainConstructionWarning,
        )

    def add_atom(self, atom, residue):
        """Add an Atom object to a Residue that has been initiated.
        Checks will be performed to determine if the atom is duplicated, 
        and DisorderedAtom object will be instantiated and filled if necessary.
        """
        name = self._assign_atom_names(atom, residue)
        if atom.altloc == " ":
            # The atom is not disordered
            residue.add(atom)
            return

        # The atom is disordered
        if residue.has_id(name):
            # The disordered atom should be already created at this point
            self._add_to_disordered_atom(atom, name, residue)
        else:
            # The residue does not contain this disordered atom
            # so we create a new one.
            self._create_disordered_atom(atom, name, residue)

    def init_atom(
        self,
        name,
        coord,
        b_factor,
        occupancy,
        altloc,
        fullname,
        serial_number=None,
        element=None,
    ):
        """Create a new Atom object.

        Arguments:
         - name - string, atom name, e.g. CA, spaces should be stripped
         - coord - Numeric array (Float0, size 3), atomic coordinates
         - b_factor - float, B factor
         - occupancy - float
         - altloc - string, alternative location specifier
         - fullname - string, atom name including spaces, e.g. " CA "
         - element - string, upper case, e.g. "HG" for mercury

        """
        # if residue is None, an exception was generated during
        # the construction of the residue
        if self.residue is None:
            return

        self.atom = Atom(
            name,
            coord,
            b_factor,
            occupancy,
            altloc,
            fullname,
            serial_number,
            element,
        )

        self.add_atom(self.atom, self.residue)

    def set_anisou(self, anisou_array):
        """Set anisotropic B factor of current Atom."""
        self.atom.set_anisou(anisou_array)

    def set_siguij(self, siguij_array):
        """Set standard deviation of anisotropic B factor of current Atom."""
        self.atom.set_siguij(siguij_array)

    def set_sigatm(self, sigatm_array):
        """Set standard deviation of atom position of current Atom."""
        self.atom.set_sigatm(sigatm_array)

    def get_structure(self):
        """Return the structure."""
        # Add the header dict
        self.structure.header = self.header
        self.structure.resolution = self.resolution
        self.structure.method = self.structure_method
        return self.structure

    def set_connect(self, connect_dict):
        """Set the connect information (disulfide bonds, etc) for atoms in each model."""
        for model in self.structure:
            model.set_connect(connect_dict)

    def set_symmetry(self, spacegroup, cell):
        """Set symmetry."""
        pass