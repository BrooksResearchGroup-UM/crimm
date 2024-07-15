"""Derived Propka Atom class to facilitate atom class conversion for pKa calculation"""
import os
import warnings
from typing import Optional, Tuple
from propka.input import read_parameter_file
from propka.parameters import Parameters
from propka.version import VersionA
from propka.lib import protein_precheck
from propka.molecular_container import MolecularContainer as _MolContnr
from propka.conformation_container import ConformationContainer
from propka.hydrogens import setup_bonding_and_protonation
from propka.atom import Atom as _ppAtom
from crimm.Modeller import TopologyGenerator
from crimm.Modeller.TopoFixer import ResidueFixer
from crimm import Data

dir_path = os.path.dirname(os.path.realpath(Data.__file__))

class PropKaAtom(_ppAtom):
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
    def __init__(self, protonator):
        """Initialize the container. This method overwrite the default
        init method from the super class"""
        self.conformation_names = []
        self.conformations = {}
        self.options = protonator # get from Protonator Class
        self.name = None
        self.version = protonator.version # same here

def _is_protonated(ph, pka):
    return pka>ph

def _is_deprotonated(ph, pka):
    return pka<ph

class PropKaProtonator:
    """PropKaProtonator class - contains all the information needed to run
    propka calculations on a protein."""

    protonation_dict = {
        'HIS': (_is_protonated, 'HSP'),
        'ASP': (_is_protonated,'ASPP'),
        'LYS': (_is_deprotonated, 'LSN'),
        'GLU': (_is_protonated, 'GLUP'),
        'CYS': (_is_deprotonated, 'CYSD'),
        'SER': (_is_deprotonated, 'SERD'),
    }
    def __init__(
            self,
            topology_loader: TopologyGenerator,
            pH: float = 7.0,
            reference: str = 'neutral',
            titrate_only: Optional[str] = None,
            thermophiles: Optional[str] = None,
            alignment: Optional[str] = None,
            mutations: Optional[str] = None,
            parameter_filename: str = 'propka.cfg',
            log_level: str = 'INFO',
            window: Tuple[float, float, float] = (0.0, 14.0, 1.0),
            grid: Tuple[float, float, float] = (0.0, 14.0, 0.1),
            mutator: Optional[str] = None,
            mutator_options: Optional[str] = None,
            display_coupled_residues: bool = False,
            reuse_ligand_mol2_file: bool = False,
            keep_protons: bool = False,
            protonate_all: bool = False,
        ) -> None:
        """Initialize the PropKaProtonator object."""
        self.pH = pH
        self.reference = reference
        self.titrate_only = titrate_only
        self.thermophiles = thermophiles
        self.alignment = alignment
        self.mutations = mutations
        self.parameter_file = parameter_filename
        self.log_level = log_level
        self.window = window
        self.grid = grid
        self.mutator = mutator
        self.mutator_options = mutator_options
        self.display_coupled_residues = display_coupled_residues
        self.reuse_ligand_mol2_file = reuse_ligand_mol2_file
        self.keep_protons = keep_protons
        self.protonate_all = protonate_all

        cfg_path = os.path.join(dir_path, self.parameter_file)
        self.parameters = read_parameter_file(cfg_path, Parameters())
        self.version = VersionA(self.parameters)
        self.mol_container = None
        self.patches = None
        self.model = None
        self.conf_container = None
        self.reportable_groups = None
        self.topo = topology_loader
        self.param = self.topo.cur_param

    def load_model(self, model):
        """Load a list of PolymerChains to propKa MolecularContainer for pKa 
        calculations. Only one conformations will be used for calculation, i.e. only 
        one altloc in the Disordered atom (the selected child) will be loaded.

        Args:
            chains: list of biopython chains

        Return:
            MolecularContainer with chain atoms loaded
        """
        self.model = model
        chains = [chain for chain in model if chain.chain_type == 'Polypeptide(L)']
        self.mol_container = MolecularContainer(self)
        conformation_name = '1A'
        self.conf_container = ConformationContainer(
            name=conformation_name,
            parameters=self.parameters,
            molecular_container=self.mol_container
        )
        self.mol_container.conformations = {conformation_name: self.conf_container}
        self.mol_container.conformation_names = [conformation_name]
        for chain in chains:
            chain.sort_residues()
            self.add_chain_to_conf_container(chain)
        
        self.mol_container.top_up_conformations()
        # make a structure precheck
        protein_precheck(
            self.mol_container.conformations, 
            self.mol_container.conformation_names
        )
        # set up atom bonding and protonation
        setup_bonding_and_protonation(self.mol_container)
        # Extract groups
        self.mol_container.extract_groups()
        # sort atoms
        self.conf_container.sort_atoms()
        # find coupled groups
        self.mol_container.find_covalently_coupled_groups()
        self.mol_container.calculate_pka()
        self._get_patch_name()

    def add_chain_to_conf_container(self, chain):
        """Add all atoms from a chain to the ConformationContainer. Terminal atom
        will be labeled accordingly."""
        pka_atoms = []
        for atom in chain.get_atoms():
            if not self.keep_protons and atom.element == 'H':
                continue
            pka_atoms.append(PropKaAtom(atom))
        # label NTER and CTER
        if pka_atoms[0].element == 'N':
            pka_atoms[0].terminal = 'N+'
        if pka_atoms[-1].element == 'O':
            pka_atoms[-1].terminal = 'C-'
        for pka_atom in pka_atoms:
            self.conf_container.add_atom(pka_atom)

    def _get_patch_name(self):
        self.patches = {}
        self.reportable_groups = {}
        for g in self.conf_container.groups:
            resname, _, chain_id = g.label.split()
            resseq = g.atom.res_num
            if chain_id not in self.patches:
                self.patches[chain_id] = {}
                self.reportable_groups[chain_id] = {}
            if g.pka_value == 0.0:
                continue
            self.reportable_groups[chain_id][resseq] = g
            if resname not in self.protonation_dict:
                continue
            eval_function, patch_name = self.protonation_dict[resname]
            if eval_function(self.pH, g.pka_value):
                self.patches[chain_id][resseq] = patch_name

    def _patch_residue(self, residue, patch_name:str):
        fixer = ResidueFixer()
        topo_def = self.topo.cur_defs[patch_name]
        if topo_def.is_patch:
            self.topo.patch_residue(residue, patch_name)
        else:
            self.topo.apply_topo_def_on_residue(residue, topo_def)
        self.param.res_def_fill_ic(residue.topo_definition, preserve=True)
        fixer.load_residue(residue)
        fixer.build_missing_atoms()
        fixer.build_hydrogens()
        fixer.remove_undefined_atoms()

    def apply_patches(self):
        """Apply patches to the model."""
        if len(self.patches) == 0:
            warnings.warn("No patches to apply.")
            return
        for chain_id, patches in self.patches.items():
            for resseq, patch_name in patches.items():
                residue = self.model[chain_id][resseq]
                self._patch_residue(residue, patch_name)
            chain = self.model[chain_id]
            if chain.topo_elements is not None:
                # Update topology elements if they are already defined
                chain.topo_elements.update()

    def report(self):
        fmt = (
            "{chain_id:2s} {i:3d} {resname:3s} pKa={pka:>5.2f} "
            "model_pKa={model_pka:>4.1f} buriedness={buried:5.3f}"
        )
        for chain_id, groups in self.reportable_groups.items():
            for i, g in groups.items():
                out_str = fmt.format(
                    chain_id = chain_id, i=i, resname=g.atom.res_name, 
                    pka=g.pka_value, model_pka = g.model_pka,
                    buried = g.buried
                )
                print(out_str)

    def to_dict(self):
        """Return a dictionary with pKa values."""
        data = {}
        for chain_id, groups in self.reportable_groups.items():
            data[chain_id] = {}
            for i, g in groups.items():
                data[chain_id][i] = {
                    'resname': g.atom.res_name, 'pka': g.pka_value, 
                    'model_pka': g.model_pka, 'buriedness': g.buried
                }
        return data
    
    def to_dataframe(self):
        """Return a pandas dataframe with pKa values."""
        data = []
        for chain_id, groups in self.reportable_groups.items():
            for i, g in groups.items():
                data.append({
                    'chain_id': chain_id, 'resseq': i, 'resname': g.atom.res_name, 
                    'pka': g.pka_value, 'model_pka': g.model_pka,
                    'buriedness': g.buried
                })
        import pandas as pd
        return pd.DataFrame(data)