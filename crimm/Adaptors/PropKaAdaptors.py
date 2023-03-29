"""Derived Propka Atom class to facilitate atom class conversion for pKa calculation"""
import os
from collections import namedtuple
from propka.input import read_parameter_file
from propka.parameters import Parameters
from propka.version import VersionA
from propka.lib import protein_precheck
from propka.molecular_container import MolecularContainer as _MolContnr
from propka.conformation_container import ConformationContainer
from propka.hydrogens import setup_bonding_and_protonation
from propka.atom import Atom as _ppAtom
# from ..Atom import Atom as _ourAtom

# Get the default options instantiated here directly so it doesn't
# need to be instantiated in any other places and get passed around
# TODO: make these options actual arguments for the main function
default_options = {
    'reference': 'neutral',
    'titrate_only': None,
    'thermophiles': None,
    'alignment': None,
    'mutations': None,
    'parameters': 'propka.cfg',
    'log_level': 'INFO',
    'pH': 7.0,
    'window': (0.0, 14.0, 1.0),
    'grid': (0.0, 14.0, 0.1),
    'mutator': None,
    'mutator_options': None,
    'display_coupled_residues': False,
    'reuse_ligand_mol2_file': False,
    'keep_protons': False,
    'protonate_all': False
}
_options = namedtuple("_options", default_options)
cur_options = _options(**default_options)
# Do the same thing for the parameters and version
dir_path = os.path.dirname(os.path.realpath(__file__))
cfg_path = os.path.join(dir_path, 'propka.cfg')
cur_parameters = read_parameter_file(cfg_path, Parameters())
cur_version = VersionA(cur_parameters)

class ProPkaAtom(_ppAtom):
    """PropKa Atom class - contains all atom information found in the PDB file


    .. versionchanged:: 3.4.0
       :meth:`make_input_line` and :meth:`get_input_parameters` have been
       removed as reading/writing PROPKA input is no longer supported.
    """

    def __init__(self, biopython_atom):
        """Initialize the Atom object. This method overwrite the default 
        init method from the super class

        Args:
            line:  Line from a PDB file to set properties of atom.
        """
        self.occ = None
        self.numb = None
        self.res_name = None
        self.type = None
        self.chain_id = None
        self.beta = None
        self.icode = None
        self.res_num = None
        self.name = None
        self.element = None
        self.x = None
        self.y = None
        self.z = None
        self.group = None
        self.group_type = None
        self.number_of_bonded_elements = {}
        self.cysteine_bridge = False
        self.bonded_atoms = []
        self.residue = None
        self.conformation_container = None
        self.molecular_container = None
        self.is_protonated = False
        self.steric_num_lone_pairs_set = False
        self.terminal = None
        self.charge = 0
        self.charge_set = False
        self.steric_number = 0
        self.number_of_lone_pairs = 0
        self.number_of_protons_to_add = 0
        self.num_pi_elec_2_3_bonds = 0
        self.num_pi_elec_conj_2_3_bonds = 0
        self.groups_extracted = 0
        self.set_properties_from_biopython_atom(biopython_atom)
        fmt = "{r.name:3s}{r.res_num:>4d}{r.chain_id:>2s}"
        self.residue_label = fmt.format(r=self)

        # ligand atom types
        self.sybyl_type = ''
        self.sybyl_assigned = False
        self.marvin_pka = False

    def set_properties_from_biopython_atom(self, bp_atom):
        """Set properties of propKa atom from a Biopython Atom.

        Args:
            bp_atom:  Biopython Atom
        """
        self.name = bp_atom.name
        self.numb = bp_atom.serial_number
        self.x, self.y, self.z = bp_atom.coord
        self.res_num = bp_atom.parent.id[1]
        self.res_name = bp_atom.parent.resname
        self.chain_id = bp_atom.parent.parent.id
        if bp_atom.parent.id[0] != ' ':
            self.type = 'hetatom'
        else:
            self.type = 'atom'
        self.occ = bp_atom.occupancy
        self.beta = bp_atom.bfactor
        self.element = bp_atom.element
        self.icode = bp_atom.parent.id[2]

class MolecularContainer(_MolContnr):
    """A derived class from MolecularContainer where the container is initiated
    with default options"""
    def __init__(self):
        """Initialize the container. This method overwrite the default
        init method from the super class"""
        self.conformation_names = []
        self.conformations = {}
        self.options = cur_options # get it directly from this script
        self.name = None
        self.version = cur_version # same here

def add_chain_to_conf_container(chain, conf_container):
    """Add all atoms from a chain to the ConformationContainer. Terminal atom
    will be labeled accordingly."""
    pka_atoms = []
    for atom in chain.get_atoms():
        # We will check the options here (hardcoded for now)
        if not cur_options.keep_protons and atom.element == 'H':
            continue
        pka_atoms.append(ProPkaAtom(atom))
    # label NTER and CTER
    if pka_atoms[0].element == 'N':
        pka_atoms[0].terminal = 'N+'
    if pka_atoms[-1].element == 'O':
        pka_atoms[-1].terminal = 'C-'
    for pka_atom in pka_atoms:
        conf_container.add_atom(pka_atom)

def convert_chains_to_mol_container(chains):
    """Load a list of PolymerChains to propKa MolecularContainer for pKa 
    calculations. Only one conformations will be used for calculation, i.e. only 
    one altloc in the Disordered atom (the selected child) will be loaded.

    Args:
        chains: list of biopython chains

    Return:
        MolecularContainer with chain atoms loaded
    """
    mol_container = MolecularContainer()
    conformation_name = '1A'
    conf_container = ConformationContainer(
        name=conformation_name,
        parameters=cur_parameters,
        molecular_container=mol_container
    )
    mol_container.conformations = {conformation_name: conf_container}
    mol_container.conformation_names = [conformation_name]
    for chain in chains:
        add_chain_to_conf_container(chain, conf_container)
    
    mol_container.top_up_conformations()
    # make a structure precheck
    protein_precheck(
        mol_container.conformations, mol_container.conformation_names
    )
    # set up atom bonding and protonation
    setup_bonding_and_protonation(mol_container)
    # Extract groups
    mol_container.extract_groups()
    # sort atoms
    conf_container.sort_atoms()
    # find coupled groups
    mol_container.find_covalently_coupled_groups()
    mol_container.calculate_pka()
    return mol_container