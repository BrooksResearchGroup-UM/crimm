"""Helpers shared by the visualization backends. No optional imports at module level."""


def rdkit_mol_to_sdf(mol, conf_id=None):
    """Return an SDF string for an RDKit molecule.

    With conf_id, only that conformer is written. Without it, every conformer
    is written as its own record, so both backends show all of them.
    """
    from rdkit import Chem

    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        return Chem.MolToMolBlock(mol) + "$$$$\n"
    if conf_id is not None:
        return Chem.MolToMolBlock(mol, confId=conf_id) + "$$$$\n"
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    return "".join(Chem.MolToMolBlock(mol, confId=i) + "$$$$\n" for i in conf_ids)


def rgb_to_hex(color):
    """Convert an RGB triple in [0, 1] to '#rrggbb'; strings pass through."""
    if isinstance(color, str):
        return color
    return "#" + "".join(f"{round(c * 255):02x}" for c in color)
