import warnings
from openbabel import openbabel as ob
from crimm.Fetchers import fetch_rcsb_as_dict
from crimm.Data.ptable import PERIODIC_TABLE

bond_order_dict = {
    # PDB mmCIF bond type names
    'SING': 1,
    'DOUB': 2,
    'TRIP': 3,
    'QUAD': 4,
    'AROM': 2, # we treat it as order of 2 instead of 1.5 here
}

def fetch_bond_info(resname):
    cifdict = fetch_rcsb_as_dict(resname)

    atom_ids = cifdict['chem_comp_atom']['atom_id']
    element_names = cifdict['chem_comp_atom']['type_symbol']
    charges = cifdict['chem_comp_atom']['charge']
    element_dict = {}
    for i, (atom_id, element_name, charge) in enumerate(
        zip(atom_ids, element_names, charges)
    ):
        element_dict[atom_id] = (i, element_name, charge)

    bond_info = cifdict['chem_comp_bond']
    a1, a2 = bond_info['atom_id_1'], bond_info['atom_id_2']
    bond_order = bond_info['value_order']
    aro_flag = [flag == 'Y' for flag in bond_info['pdbx_aromatic_flag']]
    bond_list = list(zip(a1, a2, bond_order, aro_flag))
    return bond_list

def heterogen_to_openbabel(lig):
    bond_list = fetch_bond_info(lig.resname)
    atom_ids = {}
    mol = ob.OBMol()
    mol.SetTitle(lig.resname)
    res = mol.NewResidue()
    # mol.AddResidue(res)
    mol.BeginModify()
    for i, atom in enumerate(lig, start=1):
        atom_ids[atom.name] = i
        atomic_num = PERIODIC_TABLE[atom.element]['number']
        a = mol.NewAtom()
        a.SetAtomicNum(atomic_num)
        a.SetVector(*atom.coord) # coordinates
        ob.OBAtomAssignTypicalImplicitHydrogens(a)

    arom_atoms = set()
    for a1, a2, bond_type, is_arom in bond_list:
        if a1.startswith('H') or a2.startswith('H'):
            # we add hydrogen later according to the user supplied pH value
            continue
        a1_idx, a2_idx = atom_ids[a1], atom_ids[a2]
        bond_order = bond_order_dict[bond_type]
        # AddBond uses 1-indexed atom idx
        ob_atom1, ob_atom2 = mol.GetAtom(a1_idx), mol.GetAtom(a2_idx)
        a1_hs = ob_atom1.GetImplicitHCount()-bond_order
        a2_hs = ob_atom2.GetImplicitHCount()-bond_order
        # floor to zero to avoid negtive number
        a1_hs = (abs(a1_hs)+(a1_hs))//2
        a2_hs = (abs(a2_hs)+(a2_hs))//2
        ob_atom1.SetImplicitHCount(a1_hs)
        ob_atom2.SetImplicitHCount(a2_hs)
        if is_arom:
            arom_atoms.add(a1_idx)
            arom_atoms.add(a2_idx)


    for atom_id in arom_atoms:
        atom = mol.GetAtom(atom_id)
        atom.SetAromatic(True)

    for atom_name, atom_idx in atom_ids.items():
        ob_atom = mol.GetAtom(atom_idx)
        res.AddAtom(ob_atom)
        res.SetAtomID(ob_atom, atom_name)

    mol.SetAromaticPerceived()
    mol.EndModify()

    atom_names = {v:k for k, v in atom_ids.items()}

    # Args: polaronly = false, correctForPH = false, pH = 7.4
    mol.AddHydrogens(False, True, 7.4)

    # Build hydrogen coords with SD minimization using MMFF94
    ff = ob.OBForceField.FindType('mmff94')
    success = ff.Setup(mol)
    if not success:
        warnings.warn(
            'Openbabel MMFF94 forcefield setup failed for hydrogen coord minimization!'
        )
        return
    # Constraints has to be initiated after the forcefield setup
    constraints = ob.OBFFConstraints()
    for atom in ob.OBMolAtomIter(mol):
        res.AddAtom(atom)
        atom_idx = atom.GetIdx() # 1-indexed
        if atom.GetAtomicNum() == 1:
            res.SetAtomID(atom, 'H'+str(atom_idx))
            atom.ClearCoordPtr()
        else:
            res.SetAtomID(atom, atom_names[atom_idx])
            # AddAtomConstraint uses 0-indexed atom id
            constraints.AddAtomConstraint(atom.GetId())
    res.SetName(lig.resname)
    res.SetNum(lig.id[1])
    ff.SetConstraints(constraints)
    ff.SteepestDescent(100)
    ff.GetCoordinates(mol)
    return mol
