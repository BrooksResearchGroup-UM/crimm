# Code for Sybyl atom and bond typing is from Open Drug Discovery Toolkit (ODDT)
# https://github.com/oddt/oddt/blob/master/oddt/toolkits/extras/rdkit/__init__.py
# L151
# Copyright (c) 2014, Maciej Wójcikowski
# Copyright (c) 2023, Truman Xu (徐梓乔), Brooks Lab at the University of Michigan
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of ODDT or crimm nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import requests, json
from crimm.Fetchers import fetch_rcsb_as_dict
from crimm import StructEntities
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BondType as rdBond
from rdkit.Geometry import Point3D
from crimm.Data.components_dict import TRANSITION_METALS, ALL_METALS

bond_order_dict = {
    # PDB mmCIF bond type names
    'SING': rdBond.SINGLE,
    'DOUB': rdBond.DOUBLE,
    'TRIP': rdBond.TRIPLE,
    'QUAD': rdBond.QUADRUPLE,
    'AROM': rdBond.AROMATIC,
    # probe bond type names
    'single': rdBond.SINGLE,
    'double': rdBond.DOUBLE,
    'triple': rdBond.TRIPLE,
    'quadruple': rdBond.QUADRUPLE,
    'aromatic': rdBond.AROMATIC,
}

def get_rdkit_bond_order(bo_name):
    return bond_order_dict.get(bo_name, rdBond.OTHER)

####################### ODDT code starts #######################
# Mol2 Atom typing
def _sybyl_atom_type(atom):
    """ Asign sybyl atom type
    Reference #1: http://www.tripos.com/mol2/atom_types.html
    Reference #2: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
    """
    sybyl = None
    atom_symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    hyb = atom.GetHybridization()-1  # -1 since 1 = sp, 2 = sp1 etc
    hyb = min(hyb, 3)
    degree = atom.GetDegree()
    aromtic = atom.GetIsAromatic()

    # define groups for atom types
    guanidine = '[NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])'  # strict
    # guanidine = '[NX3]([!O])([!O])!:C!:[NX3]([!O])([!O])' # corina compatible
    # guanidine = '[NX3]!@C(!@[NX3])!@[NX3,NX2]'
    # guanidine = '[NX3]C([NX3])=[NX2]'
    # guanidine = '[NX3H1,NX2,NX3H2]C(=[NH1])[NH2]' # previous
    #

    if atomic_num == 6:
        if aromtic:
            sybyl = 'C.ar'
        elif degree == 3 and _atom_matches_smarts(atom, guanidine):
            sybyl = 'C.cat'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 7:
        if aromtic:
            sybyl = 'N.ar'
        elif _atom_matches_smarts(atom, 'C(=[O,S])-N'):
            sybyl = 'N.am'
        elif degree == 3 and _atom_matches_smarts(atom, '[$(N!-*),$([NX3H1]-*!-*)]'):
            sybyl = 'N.pl3'
        elif _atom_matches_smarts(atom, guanidine):  # guanidine has N.pl3
            sybyl = 'N.pl3'
        elif degree == 4 or hyb == 3 and atom.GetFormalCharge():
            sybyl = 'N.4'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 8:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 1 and _atom_matches_smarts(atom, '[CX3](=O)[OX1H0-]'):
            sybyl = 'O.co2'
        elif degree == 2 and not aromtic:  # Aromatic Os are sp2
            sybyl = 'O.3'
        else:
            sybyl = 'O.2'
    elif atomic_num == 16:
        # http://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        if degree == 3 and _atom_matches_smarts(atom, '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'):
            sybyl = 'S.O'
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        elif _atom_matches_smarts(atom, 'S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])(-[#6])-[#6]'):
            sybyl = 'S.o2'
        else:
            sybyl = '%s.%i' % (atom_symbol, hyb)
    elif atomic_num == 15 and hyb == 3:
        sybyl = '%s.%i' % (atom_symbol, hyb)

    if not sybyl:
        sybyl = atom_symbol
    return sybyl


def _atom_matches_smarts(atom, smarts):
    idx = atom.GetIdx()
    patt = Chem.MolFromSmarts(smarts)
    for m in atom.GetOwningMol().GetSubstructMatches(patt):
        if idx in m:
            return True
    return False

def _amide_bond(bond):
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if (a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 7 or
            a2.GetAtomicNum() == 6 and a1.GetAtomicNum() == 7):
        # https://github.com/rdkit/rdkit/blob/master/Data/FragmentDescriptors.csv
        patt = Chem.MolFromSmarts('C(=O)-N')
        for m in bond.GetOwningMol().GetSubstructMatches(patt):
            if a1.GetIdx() in m and a2.GetIdx() in m:
                return True
    return False

######################## ODDT code ends ########################

tripos_bond_order_dict = {
    rdBond.SINGLE : 1,
    rdBond.DOUBLE : 2,
    rdBond.TRIPLE : 3,
    rdBond.AROMATIC : 'ar',
}

def _check_amide_bond(rdk_bond):
    if _amide_bond(rdk_bond):
        return 'am'
    return 'un'

def determine_tripos_bond_type(rdk_bond):
    return tripos_bond_order_dict.get(
        rdk_bond.GetBondType(), _check_amide_bond(rdk_bond)
    )

def _generate_mol2_title_block(resname, n_atoms, n_bonds):
    title_block = f"""# Generated by crimm with rdkit
@<TRIPOS>MOLECULE
{resname}
{n_atoms} {n_bonds} 1
SMALL
GASTEIGER
****

"""
    return title_block

def _determine_ligname(mol):
    atom = list(mol.GetAtoms())[0]
    if info := atom.GetPDBResidueInfo():
        ligname = info.GetResidueName()
    elif mol.HasProp('Description'):
        ligname = mol.GetProp('Description')
    elif mol.HasProp('_Name'):
        ligname = mol.GetProp('_Name')
    else:
        ligname = 'LIG'
    return ligname.upper()[:3]

def MolToMol2File(mol, ligname = None, filename = None):
    """Write a mol2 file from a RDKit Mol Object."""
    if ligname is None:
        ligname = _determine_ligname(mol)

    pos = mol.GetConformer().GetPositions()
    atoms = list(mol.GetAtoms())

    tripos_atom_format = '{:<6} {:<5} {:>7.3f} {:>7.3f} {:>7.3f} {:<6} {} {} {:>6.3f}'
    tripos_bond_format = '{:<5} {:<5} {:<5} {:<2}'
    atom_lines = []
    bond_lines = []
    for i, (atom, coords) in enumerate(zip(atoms, pos), start=1):
        if pdbinfo := atom.GetPDBResidueInfo():
            name = pdbinfo.GetName()
        else:
            name = atom.GetSymbol()+str(i)
        sybyl_type = _sybyl_atom_type(atom)
        charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_lines.append(
            tripos_atom_format.format(
                i, name, *coords, sybyl_type, 1, ligname, charge
            )
        )
        

    tripos_bonds = []
    for bond in mol.GetBonds():
        st = bond.GetBeginAtomIdx()+1
        end = bond.GetEndAtomIdx()+1
        ordered_ids = sorted((st, end))
        bo = determine_tripos_bond_type(bond)
        tripos_bonds.append((*ordered_ids, bo))

    for i, bond_info in enumerate(sorted(tripos_bonds), start=1):
        bond_lines.append(tripos_bond_format.format(i, *bond_info))

    n_atoms, n_bonds = len(atom_lines), len(bond_lines)
    title_block = _generate_mol2_title_block(ligname, n_atoms, n_bonds)

    if filename is None:
        filename = f'{ligname}.mol2'
    with open(filename, 'w', encoding='UTF-8') as f:
        f.write(title_block)
        f.write('@<TRIPOS>ATOM\n')
        for l in atom_lines:
            f.write(l+'\n')
        f.write('@<TRIPOS>BOND\n')
        for l in bond_lines:
            f.write(l+'\n')
        f.write(f'@<TRIPOS>SUBSTRUCTURE\n1 {ligname} 1')


class LigandBondOrderException(Exception):
    """Define class LigandBondOrderException."""

class SmilesQueryException(Exception):
    """Define class SmilesQueryException."""

class LigandBondOrderWarning(Warning):
    """Define class LigandBondOrderException."""

class SmilesQueryWarning(Warning):
    """Define class SmilesQueryException."""

class RDKitHetConverter:
    """Convert a crimm heterogen object to an rdkit mol object. The mol object 
    WILL have the correct bond orders. Upon loading the heterogen, the CIF 
    definition is queried from rcsb.org, which contains all the necessary
    information to create the rdkit mol object. The mol object is then used to
    generate a mol2 file.

    Attributes:
        lig (StructEntities.Heterogen): The heterogen object to be converted.
        lig_pdbid (str): The PDB ID of the heterogen.
        chain_id (str): The chain ID of the heterogen.
        element_dict (dict): A dictionary mapping the atom name to the element
            symbol.
        bond_dict (list): A list of tuples containing the bond information.
        resname (str): The residue name of the heterogen.
        resnum (int): The residue number of the heterogen.
        mol (rdkit.Chem.rdchem.Mol): The constructed rdkit mol object.
    
    Raises:
        TypeError: If the input is not a crimm heterogen object constructed from
        pdb mmcif file.
    
    Methods:
        load_heterogen: Load a heterogen object and create the rdkit mol object.
        get_mol: Return the rdkit mol object.
        write_mol2: Write the mol2 file of the loaded heterogen.
    
    Examples:
        >>> from crimm import fetch_rcsb
        >>> from crimm.Adaptors import RDKitHetConverter
        >>> structure = fetch_rcsb('1aka')
        ### Select the heterogen from the structure
        ### Note that the heterogen is at residue level not chain level
        >>> lig = structure[1]['D'].residues[0]
        >>> rdk_converter = RDKitHetConverter()
        >>> rdk_converter.load_heterogen(lig)
        >>> mol = rdk_converter.get_mol()
        >>> rdk_converter.write_mol2()
    """
    def __init__(self):
        self.lig = None
        self.lig_pdbid = None
        self.chain_id = None
        self.element_dict = None
        self.bond_dict = None
        self.resname = None
        self.resnum = None
        self._edmol = Chem.EditableMol(Chem.Mol())
        self._pending_hydrogens = None
        self._rd_atoms = None
        self._sanitize = True # determine if sanitization can be performed
        self.mol = None

    def load_heterogen(self, lig):
        """Load a heterogen object and create the rdkit mol object."""
        if isinstance(lig, StructEntities.Heterogen):
            self.lig = lig
        else:
            raise TypeError("Input must be a Heterogen object (Residue level).")

        self.lig_pdbid = lig.resname
        self.chain_id = lig.parent.id
        self.resname = lig.resname
        self.resnum = lig.id[1]
        self._edmol = Chem.EditableMol(Chem.Mol())
        self._pending_hydrogens = None
        self._rd_atoms = None
        self._sanitize = True
        self.mol = None

        cifdict = fetch_rcsb_as_dict(self.lig_pdbid)

        atom_ids = cifdict['chem_comp_atom']['atom_id']
        element_names = cifdict['chem_comp_atom']['type_symbol']
        charges = cifdict['chem_comp_atom']['charge']
        self.element_dict = {}
        for i, (atom_id, element_name, charge) in enumerate(
            zip(atom_ids, element_names, charges)
        ):
            if element_name.capitalize() in ALL_METALS:
                # if the heterogen contains a metal, we do not perform sanitization
                self._sanitize = False
            self.element_dict[atom_id] = (i, element_name, charge)

        bond_info = cifdict['chem_comp_bond']
        a1, a2 = bond_info['atom_id_1'], bond_info['atom_id_2']
        bond_order = bond_info['value_order']
        aro_flag = [flag == 'Y' for flag in bond_info['pdbx_aromatic_flag']]
        self.bond_dict = list(zip(a1, a2, bond_order, aro_flag))

    def _create_rdkit_PDBinfo(self, pdb_atom_name, altloc):
        serial_number = self.element_dict[pdb_atom_name][0]
        pdb_info = Chem.AtomPDBResidueInfo(
            pdb_atom_name, serial_number, altloc, residueName = self.resname,
            residueNumber = self.resnum, chainId = self.chain_id,
            isHeteroAtom = True
        )
        return pdb_info
    
    def _create_rdkit_atoms(self):
        self._rd_atoms = {}
        for atom in self.lig.atoms:
            rd_atom = Chem.Atom(atom.element.capitalize())
            pdb_info = self._create_rdkit_PDBinfo(atom.name, atom.altloc)
            rd_atom.SetPDBResidueInfo(pdb_info)
            charge = self.element_dict[atom.name][2]
            element = self.element_dict[atom.name][1].capitalize()
            rd_atom.SetFormalCharge(charge)
            atom_idx = self._edmol.AddAtom(rd_atom)
            self._rd_atoms[atom.name] = atom_idx

    def _add_rdkit_bonds(self):
        self._pending_hydrogens = {}
        for a1, a2, bo_name, is_arom in self.bond_dict:
            elem1 = self.element_dict[a1][1].capitalize()
            elem2 = self.element_dict[a2][1].capitalize()
            n_transition_metals = (
                elem1 in TRANSITION_METALS
            ) + (
                elem2 in TRANSITION_METALS
            )
            if a1 not in self._rd_atoms:
                if a2 not in self._pending_hydrogens:
                    self._pending_hydrogens[a2] = []
                self._pending_hydrogens[a2].append(a1)
                if self.element_dict[a1][1] != 'H':
                    raise ValueError(
                        'CIF definition does not match the heterogen instance! '
                        f'Fail to find atom {a1} in mol {self.resname}'
                    )
                continue
            elif a2 not in self._rd_atoms:
                if a1 not in self._pending_hydrogens:
                    self._pending_hydrogens[a1] = []
                self._pending_hydrogens[a1].append(a2)
                if self.element_dict[a2][1] != 'H':
                    raise ValueError(
                        'CIF definition does not match the heterogen instance! '
                        f'Fail to find atom {a2} in mol {self.resname}'
                    )
                continue
            if n_transition_metals == 1:
                # we assume metal-ligand bond is a dative bond 
                bo = rdBond.DATIVE
            else:
                bo = get_rdkit_bond_order(bo_name)
            idx1, idx2 = self._rd_atoms[a1], self._rd_atoms[a2]
            self._edmol.AddBond(idx1, idx2, bo)

    def _create_conformer(self):
        conf = Chem.Conformer()
        conf.Set3D(True)
        for atom in self.lig:
            idx = self._rd_atoms[atom.name]
            coord = Point3D(*atom.coord)
            conf.SetAtomPosition(idx, coord)
        return conf

    def _add_hydrogen_PDBinfo(self, mol):
        for heavy_atom_name, hydrogen_names in self._pending_hydrogens.items():
            heavy_atom = mol.GetAtomWithIdx(self._rd_atoms[heavy_atom_name])
            nei_hs = []
            for nei in heavy_atom.GetNeighbors():
                if nei.GetAtomicNum() == 1:
                    nei_hs.append(nei)
            for hydrogen_name, atom in zip(hydrogen_names, nei_hs):
                pdb_info = self._create_rdkit_PDBinfo(hydrogen_name, '')
                atom.SetPDBResidueInfo(pdb_info)

    def _create_rdkit_mol(self):
        self._create_rdkit_atoms()
        self._add_rdkit_bonds()
        mol = self._edmol.GetMol()
        conf = self._create_conformer()
        mol.AddConformer(conf)
        if not self._sanitize:
            warnings.warn(
                'The heterogen contains metal atoms! Sanitization is not performed.'
                'Hygrogen atoms are not added to the mol object.'
            )
            self.mol = mol
            return
        AllChem.SanitizeMol(mol)
        mol = AllChem.AddHs(mol, addCoords=True)
        AllChem.SanitizeMol(mol)
        AllChem.ComputeGasteigerCharges(mol)
        self._add_hydrogen_PDBinfo(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        self.mol = mol

    def get_mol(self):
        """Return the rdkit mol object."""
        if self.mol is None:
            self._create_rdkit_mol()
        return self.mol

    def write_mol2(self, filename = None):
        """Write the mol2 file of the loaded heterogen."""
        MolToMol2File(self.mol, self.resname, filename)

def _build_smiles_query(resname):
    query = '''
    {
        chem_comps(comp_ids: ["{var_lig_id}"]) {
            rcsb_id
            rcsb_chem_comp_descriptor {
            SMILES
            }
        }
    }
    '''.replace("{var_lig_id}", resname)
    return query

def query_rcsb_for_smiles(het_res):
    """Query the canonical SMILES for the heterogen molecule from PDB based on chem_comps
    ID"""
    url="https://data.rcsb.org/graphql"
    response = requests.post(
            url, json={'query': _build_smiles_query(het_res.resname)}, timeout=1000
        )
    response_dict = json.loads(response.text)
    return_vals = response_dict['data']['chem_comps']
    if not return_vals:
        warnings.warn(
            f'Query on {het_res.resname} did not return requested information: '
            'chem_comps (smiles string)'
        )
        return

    smiles = return_vals[0]['rcsb_chem_comp_descriptor']['SMILES']
    return smiles

def heterogen_to_rdkit(het_res, smiles=None):
    """Convert a heterogen to an rdkit mol. The bond orders are set based on the
    SMILES string. If the SMILES string is not provided, the function will query
    the RCSB PDB for the SMILES string.
    
    Args:
        het_res (Heterogen): The heterogen to be converted.
        smiles (str, optional): The SMILES string of the heterogen. Defaults to None.
        
    Returns:
        rdkit.Chem.rdchem.Mol: The rdkit mol of the heterogen.
    """
    mol = Chem.MolFromPDBBlock(get_pdb_str(het_res))
    if smiles is None:
        smiles = query_rcsb_for_smiles(het_res)
    # We do not allow rdkit mol return if the correct bond orders are not set
    if smiles is None:
        msg = (
            'Fail to set bond orders on the Ligand mol! PDB query on SMILES does not'
            'return any result.'
        )
        warnings.warn(msg, SmilesQueryWarning)
        return

    template = AllChem.MolFromSmiles(smiles)
    template = Chem.RemoveHs(template)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
        if het_res.pdbx_description is not None:
            mol.SetProp('Description', str(het_res.pdbx_description))
        return mol

    except ValueError:
        msg = (
            'No structure match found! Possibly the SMILES string supplied or reported on PDB '
            'mismatches the ligand structure.'
        )
        warnings.warn(msg, LigandBondOrderWarning)
        return

def create_probe_mol(probe, use_conf=True):
    edmol = Chem.EditableMol(Chem.Mol())
    rd_atoms = {}
    conf = Chem.Conformer()
    conf.Set3D(True)
    for atom in probe.atoms:
        rd_atom = Chem.Atom(atom.element.capitalize())
        atom_idx = edmol.AddAtom(rd_atom)
        rd_atoms[atom.name] = atom_idx
        coord = Point3D(*atom.coord)
        conf.SetAtomPosition(atom_idx, coord)
    for bond in probe.bonds:
        a1, a2 = bond[0].name, bond[1].name
        idx1, idx2 = rd_atoms[a1], rd_atoms[a2]
        bo = get_rdkit_bond_order(bond.type)
        edmol.AddBond(idx1, idx2, bo)
    mol = edmol.GetMol()
    if use_conf:
        mol.AddConformer(conf)
    # AllChem.SanitizeMol(mol)
    # mol = AllChem.AddHs(mol, addCoords=use_conf)
    # AllChem.SanitizeMol(mol)
    # AllChem.ComputeGasteigerCharges(mol)
    # Chem.AssignStereochemistryFrom3D(mol)
    return mol

def create_probe_confomers(probe, conf_coords: np.ndarray):
    if conf_coords.shape[1] != len(probe.atoms):
        raise ValueError(
            'Number of atoms in probe and number of coordinates do not match!'
        )
    mol = create_probe_mol(probe, use_conf=False)
    for conf_coord in conf_coords:
        conf = Chem.Conformer()
        conf.Set3D(True)
        for atom_idx, coord in enumerate(conf_coord):
            coord = Point3D(*coord.astype(np.float64))
            conf.SetAtomPosition(atom_idx, coord)
        mol.AddConformer(conf, assignId=True)
    return mol

def write_conformers_sdf(mol, filename):
    conformers = mol.GetConformers()
    if not conformers:
        raise ValueError('No conformers found in the molecule!')
    with AllChem.SDWriter(filename) as writer:
        for conf in conformers:
            writer.write(mol, confId=conf.GetId())