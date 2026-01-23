import tempfile
import warnings
import numpy as np
from pathlib import Path

import pycharmm as pcm
from pycharmm import read, psf, coor
from pycharmm import generate
import pycharmm.settings as pcm_settings
from pycharmm import ic, cons_harm, cons_fix
from pycharmm import energy
from pycharmm import minimize as _minimize
from pycharmm.generate import patch as charmm_patch
from pycharmm.psf import get_natom, delete_atoms
from Bio.PDB.Selection import unfold_entities
from crimm.IO.PDBString import get_pdb_str
from crimm.IO import write_psf, write_crd
from crimm.StructEntities.Residue import Heterogen
from crimm.Data.components_dict import nucleic_letters_1to3

def empty_charmm():
    """If any atom exists in current CHARMM runtime, remove them."""
    if get_natom() > 0:
        delete_atoms()


def _entity_has_lonepairs(entity) -> bool:
    """Check if entity contains any residues with lone pairs (e.g., CGENFF ligands)."""
    res_list = unfold_entities(entity, 'R')
    for res in res_list:
        if isinstance(res, Heterogen) and len(res.lone_pairs) > 0:
            return True
    return False


def _load_psf_crd(entity, append=False, separate_crystal_segids=False):
    """Load an entity into pyCHARMM using PSF/CRD format.

    This is the core helper that writes temp PSF/CRD files and loads them.

    Parameters
    ----------
    entity : Model, Chain, or Residue
        The entity to load into CHARMM
    append : bool, default False
        If True, append to existing PSF in CHARMM

    Raises
    ------
    RuntimeError
        If PSF or CRD files are not written correctly
    """
    import os
    psf_path = None
    crd_path = None
    try:
        # Create temp files with delete=False for cross-platform compatibility
        with tempfile.NamedTemporaryFile('w', suffix='.psf', delete=False) as psf_f:
            psf_path = psf_f.name
        with tempfile.NamedTemporaryFile('w', suffix='.crd', delete=False) as crd_f:
            crd_path = crd_f.name

        # Write PSF and CRD files (these functions handle their own file I/O)
        write_psf(entity, psf_path, separate_crystal_segids=separate_crystal_segids)
        write_crd(entity, crd_path)

        # Validate files were written correctly
        if not os.path.exists(psf_path) or os.path.getsize(psf_path) == 0:
            raise RuntimeError(f"PSF file was not written correctly: {psf_path}")
        if not os.path.exists(crd_path) or os.path.getsize(crd_path) == 0:
            raise RuntimeError(f"CRD file was not written correctly: {crd_path}")

        # Load into pyCHARMM
        read.psf_card(psf_path, append=append)
        # When appending PSF, also use append for CRD to offset atom indices
        read.coor_card(crd_path, append=append)
    finally:
        # Clean up temp files
        if psf_path and os.path.exists(psf_path):
            os.unlink(psf_path)
        if crd_path and os.path.exists(crd_path):
            os.unlink(crd_path)

    # Handle lone pairs if present (CGENFF ligands)
    if _entity_has_lonepairs(entity):
        print("[crimm] Creating lone pair coordinates using COOR SHAKE")
        pcm.lingo.charmm_script("coor shake")


def load_topology(topo_generator):
    """Load topology and parameter files from a TopoGenerator object."""
    load_water_ions = False
    load_cgenff_ligand = False
    if 'cgenff' in topo_generator.res_def_dict:
        load_cgenff_ligand = True
        # rearrange the order of cgenff
        topo_generator.res_def_dict['cgenff'] = topo_generator.res_def_dict.pop('cgenff')
    if 'cgenff' in topo_generator.param_dict:
        topo_generator.param_dict['cgenff'] = topo_generator.param_dict.pop('cgenff')

    for i, (topo_type, topo_loader) in enumerate(topo_generator.res_def_dict.items()):
        if topo_type == 'water_ions':
            load_water_ions = True
            continue
        if topo_type == 'cgenff':
            continue
        with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
            tf.write(f'* {topo_type.upper()} RTF Loaded from crimm\n')
            for line in topo_loader._raw_data_strings:
                if line.upper().startswith('RESI') or line.upper().startswith('PRES'):
                    line = '\n'+line
                tf.write(line+'\n')
            tf.flush() # has to flush first for long files!
            read.rtf(tf.name, append = bool(i))

    for i, (param_type, param_loader) in enumerate(topo_generator.param_dict.items()):
        if param_type == 'water_ions':
            load_water_ions = True
            continue
        if param_type == 'cgenff':
            continue
        with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
            tf.write(f'* {param_type.upper()} PRM Loaded from crimm\n')
            for line in param_loader._raw_data_strings:
                tf.write(line+'\n')
            tf.flush()
            read.prm(tf.name, append = bool(i), flex=True)

    # load cgenff.rtf and cgenff.prm after all bio polymers
    if load_cgenff_ligand:
        load_cgenff_toppar()
        ligandrtf_blocks = topo_generator.cgenff_loader.toppar_blocks
        for resname, data_block in ligandrtf_blocks.items():
            with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
                tf.write(f'* CGENFF TOPPAR for {resname} Loaded from crimm\n')
                tf.write(data_block)
                tf.flush() # has to flush first for long files!
                read.stream(tf.name)
        
    # load water_ions.str at the end
    if load_water_ions:
        load_solvent_toppar()

def load_solvent_toppar():
    """Load default solvent model (TIP3 water + ions) into pyCHARMM."""
    abs_path = Path(__file__).resolve().parent.parent
    abs_path = abs_path / "Data/toppar/water_ions.str"
    with open(abs_path, 'r', encoding='utf-8') as f:
        with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
            tf.write('* WATER ION TOPPAR Loaded from crimm\n')
            for line in f.readlines():
                tf.write(line)
            tf.flush()
            read.stream(tf.name)

def load_cgenff_toppar():
    """Load default CGENFF parameters into pyCHARMM."""
    abs_path = Path(__file__).resolve().parent.parent
    rtf_abs_path = abs_path / "Data/toppar/cgenff.rtf"
    prm_abs_path = abs_path / "Data/toppar/cgenff.prm"
    with open(rtf_abs_path, 'r', encoding='utf-8') as f:
        with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
            tf.write('* CGENFF RTF Loaded from crimm\n')
            tf.write(f.read())
            tf.flush() # has to flush first for long files!
            read.rtf(tf.name, append = True)

    pcm_settings.set_bomb_level(-1)
    with open(prm_abs_path, 'r', encoding='utf-8') as f:
        with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
            tf.write('* CGENFF PRM Loaded from crimm\n')
            tf.write(f.read())
            tf.flush() # has to flush first for long files!
            read.prm(tf.name, append=True, flex=True)
    pcm_settings.set_bomb_level(0)

def load_chain(chain, hbuild=False, report=False, use_psf_crd=True, append=False):
    """Load a protein/nucleic chain into pyCHARMM.

    Parameters
    ----------
    chain : PolymerChain
        The chain to load (must have topology generated via TopologyGenerator)
    hbuild : bool, default False
        Run HBUILD for hydrogens (only used for legacy PDB mode)
    report : bool, default False
        Print IC table (only used for legacy PDB mode)
    use_psf_crd : bool, default True
        If True (default), use PSF/CRD format - simpler and recommended.
        If False, use deprecated PDB-based loading with sequence generation.
    append : bool, default False
        If True, append to existing PSF in CHARMM (for incremental loading).

    Returns
    -------
    str
        The segment ID used for this chain
    """
    if not chain.is_continuous():
        raise ValueError("Chain is not continuous! Fix the chain first!")

    # Determine segment ID based on chain type
    if chain.chain_type == 'Polyribonucleotide':
        segid = f'NUC{chain.id[0]}'
    else:
        segid = f'PRO{chain.id[0]}'

    # Set segid on all residues
    for res in chain:
        res.segid = segid

    if use_psf_crd:
        # PSF/CRD approach (default) - simpler and recommended
        # Topology is already in the PSF, no generation/patching needed
        _load_psf_crd(chain, append=append)
        return segid
    else:
        # Deprecated PDB-based implementation
        warnings.warn(
            "PDB-based loading (use_psf_crd=False) is deprecated and will be removed "
            "in a future version. Use PSF/CRD format (default) for simpler and more "
            "reliable loading.",
            DeprecationWarning,
            stacklevel=2
        )
        m_chain = _get_charmm_named_chain(chain, segid)
        first_res = m_chain.child_list[0]
        last_res = m_chain.child_list[-1]
        first_patch, last_patch = '', ''
        other_patches = {}
        for res in m_chain.residues[1:-1]:
            # iterate over all residues except the first and last to find 
            # if any residue has been patched
            if (patch_name:=res.topo_definition.patch_with) is not None:
                if patch_name == 'DISU':
                    # skip disulfide bond patch in this step
                    continue
                resseq = res.id[1]
                # CHARMM requires the patch location to be in the format of
                # SEGID RESSEQ
                patch_loc = f'{segid} {resseq}'
                other_patches[patch_loc] = patch_name
        if (patch_name:=first_res.topo_definition.patch_with) is not None:
            first_patch = patch_name
            
        if (patch_name:=last_res.topo_definition.patch_with) is not None:
            last_patch = patch_name
            
        with tempfile.NamedTemporaryFile('w') as tf:
            tf.write(get_pdb_str(m_chain, use_charmm_format=True))
            tf.write('END\n')
            tf.flush()
            _load_chain(
                tf.name, segnames=[segid], first_patch=first_patch, last_patch=last_patch,
                hbuild=hbuild, report=report
            )
        if len(other_patches) > 0:
            for patch_loc, patch_name in other_patches.items():
                resseq = int(patch_loc.split()[1])
                charmm_patch(patch_name, patch_sites=patch_loc)
                pcm.lingo.charmm_script(
                    f'hbuild sele segid {segid} .and. resi {resseq} '
                    '.and. hydrogen end'
                )
        return segid

def load_ligands(ligand_chains, segids=None, use_psf_crd=True, append=False):
    """Load a list of ligand chains into pyCHARMM.

    Parameters
    ----------
    ligand_chains : list
        List of ligand chains (each chain may contain multiple ligand residues)
    segids : list, optional
        Segment IDs for each ligand. If None, auto-generates LG00, LG01, etc.
    use_psf_crd : bool, default True
        If True (default), use PSF/CRD format - simpler and recommended.
        If False, use deprecated PDB-based loading.
    append : bool, default False
        If True, append to existing PSF in CHARMM (for incremental loading).

    Returns
    -------
    list
        The segment IDs used for each ligand
    """
    all_ligands = [res for chain in ligand_chains for res in chain]
    if segids is None:
        segids = [f'LG{i:02d}' for i in range(len(all_ligands))]
    elif len(segids) != len(all_ligands):
        raise ValueError("Number of segids must match number of ligands")

    if use_psf_crd:
        # PSF/CRD approach (default) - simpler and recommended
        for i, (segid, lig_res) in enumerate(zip(segids, all_ligands)):
            lig_res.segid = segid
            print(f"[crimm] Loading ligand {lig_res.resname} SEG: {segid}")
            # First ligand uses caller's append value, subsequent ones always append
            should_append = append if i == 0 else True
            _load_psf_crd(lig_res, append=should_append)
    else:
        # Deprecated PDB-based implementation
        warnings.warn(
            "PDB-based loading (use_psf_crd=False) is deprecated and will be removed "
            "in a future version. Use PSF/CRD format (default) for simpler and more "
            "reliable loading.",
            DeprecationWarning,
            stacklevel=2
        )
        for segid, lig_res in zip(segids, all_ligands):
            lig_res.segid = segid
            print(f"[crimm] Loading ligand {lig_res.resname} SEG: {segid}")
            with tempfile.NamedTemporaryFile('w') as tf:
                tf.write(get_pdb_str(lig_res, use_charmm_format=True))
                tf.write('END\n')
                tf.flush()
                read.sequence_pdb(tf.name)
                generate.new_segment(
                    seg_name=segid,
                    first_patch='',
                    last_patch='',
                    angle=True,
                    dihedral=True
                )
                read.pdb(tf.name, resid=True)
        
        lone_pair_ligands = [lig.resname for lig in all_ligands if len(lig.lone_pairs) > 0]

        if len(lone_pair_ligands) > 0:
            print(
                "[crimm] Creating lone pair coordinates for ligands "
                f"{','.join(lone_pair_ligands)} using CHARMM command COOR SHAKE"
            )
            pcm.lingo.charmm_script("coor shake")

    return segids

def load_water(water_chains, segids=None, use_psf_crd=True, append=False):
    """Load water chains into pyCHARMM.

    Parameters
    ----------
    water_chains : list
        List of water chains to load
    segids : list, optional
        Segment IDs for each water chain. If None, auto-generates WT00, WT01, etc.
    use_psf_crd : bool, default True
        If True (default), use PSF/CRD format - simpler and recommended.
        If False, use deprecated PDB-based loading.
    append : bool, default False
        If True, append to existing PSF in CHARMM (for incremental loading).

    Returns
    -------
    list
        The segment IDs used for each water chain
    """
    # Currently only supports TIP3 water model
    if segids is None:
        segids = [f'WT{i:02d}' for i in range(len(water_chains))]
    elif len(segids) != len(water_chains):
        raise ValueError("Number of segids must match number of water chains")

    if use_psf_crd:
        # PSF/CRD approach (default) - simpler and recommended
        for i, chain in enumerate(water_chains):
            # Segment ID for water chains is defined by PSFWriter based on water source
            should_append = append if i == 0 else True
            segid = chain.residues[0].segid
            _load_psf_crd(chain, append=should_append)
    else:
        # Deprecated PDB-based implementation
        warnings.warn(
            "PDB-based loading (use_psf_crd=False) is deprecated and will be removed "
            "in a future version. Use PSF/CRD format (default) for simpler and more "
            "reliable loading.",
            DeprecationWarning,
            stacklevel=2
        )
        for segid, chain in zip(segids, water_chains):
            for res in chain:
                res.segid = segid
            print(f"[crimm] Loading water chain {segid}")
            chain = chain.copy()
            chain.reset_atom_serial_numbers(reset_current_only=True)
            for res in chain:
                res.resname = res.topo_definition.resname
            with tempfile.NamedTemporaryFile('w') as tf:
                tf.write(get_pdb_str(chain, use_charmm_format=True, reset_serial=False))
                tf.write('END\n')
                tf.flush()

                read.sequence_pdb(tf.name)
                generate.new_segment(
                    seg_name=segid,
                    first_patch='',
                    last_patch='',
                    angle=False,
                    dihedral=False
                )
                read.pdb(tf.name, resid=True)

    return segids

def load_ions(ion_chains, use_psf_crd=True, append=False):
    """Load ion chains into pyCHARMM.

    Parameters
    ----------
    ion_chains : list
        List of ion chains to load
    use_psf_crd : bool, default True
        If True (default), use PSF/CRD format - simpler and recommended.
        If False, use deprecated PDB-based loading.
    append : bool, default False
        If True, append to existing PSF in CHARMM (for incremental loading).

    Returns
    -------
    list
        The segment IDs used for each ion chain
    """
    segids = []
    
    if use_psf_crd:
        # PSF/CRD approach (default) - simpler and recommended
        for i, chain in enumerate(ion_chains):
            # Segment ID for ion chains is defined by PSFWriter based on ion source
            should_append = append if i == 0 else True
            _load_psf_crd(chain, append=should_append)
            segid = chain.residues[0].segid
            segids.append(segid)
    else:
        # Deprecated PDB-based implementation
        warnings.warn(
            "PDB-based loading (use_psf_crd=False) is deprecated and will be removed "
            "in a future version. Use PSF/CRD format (default) for simpler and more "
            "reliable loading.",
            DeprecationWarning,
            stacklevel=2
        )
        for i, chain in enumerate(ion_chains):
            segid = f'IO{i:02d}'
            print(f"[crimm] Loading ion chain {segid}")
            for res in chain:
                res.segid = segid
            # we need to copy the chain here, since we might modify atom_serial_number
            chain = chain.copy()
            chain.reset_atom_serial_numbers(reset_current_only=True)
            
            with tempfile.NamedTemporaryFile('w') as tf:
                tf.write(get_pdb_str(chain, use_charmm_format=True, reset_serial=False))
                tf.write('END\n')
                tf.flush()

                read.sequence_pdb(tf.name)
                generate.new_segment(
                    seg_name=segid,
                    first_patch='',
                    last_patch='',
                    angle=False,
                    dihedral=False
                )
                read.pdb(tf.name, resid=True)
            segids.append(segid)

    return segids


def load_model(model, use_psf_crd=True, load_params=True, separate_crystal_segids=False):
    """Load an entire OrganizedModel into pyCHARMM.

    This is a convenience function that loads all components of a model
    (protein chains, ligands, water, ions) in a single call.

    Parameters
    ----------
    model : OrganizedModel
        Model with topology generated via TopologyGenerator
    use_psf_crd : bool, default True
        If True (default), load entire model via single PSF/CRD call - most efficient.
        If False, load components individually using legacy PDB-based approach.
    load_params : bool, default True
        If True (default), also load topology parameters (RTF/PRM) via load_topology().
    seperate_crystal_segids : bool, default False
        If True, separate crystal segids in the PSF file that is loaded into CHARMM. This
        will not affect segids that are already assigned to the residues in the model.

    Notes
    -----
    When use_psf_crd=True, the entire model is loaded at once via PSF/CRD.
    This is simpler and preserves all topology (including disulfide bonds) automatically.
    
    When use_psf_crd=False, components are loaded individually using the deprecated
    PDB-based loading and requires separate patch_disu_from_model() call.

    Examples
    --------
    >>> from crimm.Adaptors.pyCHARMMAdaptors import load_model, load_topology
    >>> # Load everything in one call (recommended)
    >>> load_model(model)
    >>>
    >>> # Or load topology separately and just load structure
    >>> load_topology(model.topology_loader)
    >>> load_model(model, load_params=False)
    """
    if load_params:
        load_topology(model.topology_loader)

    if use_psf_crd:
        # PSF/CRD approach (default) - single call for entire model
        # Topology is already in the PSF, including disulfides - no patching needed
        _load_psf_crd(model, append=False, separate_crystal_segids=separate_crystal_segids)
    else:
        # Deprecated: load components separately using PDB-based approach
        warnings.warn(
            "PDB-based loading (use_psf_crd=False) is deprecated and will be removed "
            "in a future version. Use PSF/CRD format (default) for simpler and more "
            "reliable loading.",
            DeprecationWarning,
            stacklevel=2
        )
        # Load protein chains
        for chain in model.protein:
            load_chain(chain, use_psf_crd=False)
        
        # Load ligands (including phosphorylated ligands and co-solvents)
        all_ligand_chains = model.ligand + model.phos_ligand + model.co_solvent
        if all_ligand_chains:
            load_ligands(all_ligand_chains, use_psf_crd=False)
        
        # Load water
        if model.solvent:
            load_water(model.solvent, use_psf_crd=False)
        
        # Load ions
        if model.ion:
            load_ions(model.ion, use_psf_crd=False)
        
        # Apply disulfide patches (only needed for PDB-based loading)
        patch_disu_from_model(model)

def _get_charmm_named_chain(chain, segid):
    for res in chain:
        res.segid = segid
    m_chain = chain.copy()
    if chain.chain_type == 'Polypeptide(L)':
        for res in m_chain:
            if res.resname == 'HIS':
                # we uses HSD for default HIS protonation state here
                res.resname = res.topo_definition.resname
    elif chain.chain_type == 'Polyribonucleotide':
        for res in m_chain:
            res.resname = nucleic_letters_1to3[res.resname]
    return m_chain


def _load_chain(
        pdb_filepath, segnames, hbuild = True, report = False,
        first_patch='', last_patch=''
    ):
     
    read.sequence_pdb(pdb_filepath)

    for seg in segnames:
        generate.new_segment(
            seg_name=seg, 
            first_patch=first_patch,
            last_patch=last_patch,
            setup_ic=True,
        )

    read.pdb(pdb_filepath, resi=True)

    ic.prm_fill(replace_all=False)
    ic.build()

    if report:
        pcm.lib.charmm.ic_print()
    if hbuild:
        pcm.lingo.charmm_script("hbuild sele type H* end")

def patch_disu_from_model(model):
    """Patch disulfide bonds found in a model object."""
    if 'disulf' in model.connect_dict:
        for res1, res2 in model.connect_dict['disulf']:
            seg1, seg2 = res1['chain'], res2['chain']
            seq1, seq2 = res1['resseq'], res2['resseq']
            patch_arg = f'PRO{seg1} {seq1} PRO{seg2} {seq2}'
            print('[Excuting CHARMM Command] patch DISU', patch_arg)
            generate.patch('DISU', patch_arg)

def ok_to_sync(chain):
    """DEPRECATED: Use fetch_coords_from_charmm instead."""
    resname_list = []
    if chain.undefined_res:
        print(f"Undefined residue exists! {chain.undefined_res}")
        return False
    for res in chain:
        if res.topo_definition is not None:
            resname_list.append(res.topo_definition.resname)
        else:
            return False
    return resname_list == psf.get_res()

def sync_coords(chain):
    """DEPRECATED: Use fetch_coords_from_charmm instead."""
    if not ok_to_sync(chain):
        print("[crimm] ABORT: Possible residue sequences mismatch!")
        return
    ibase = list(psf.get_ibase())
    new_coord_df = coor.get_positions()
    atom_coords = list(zip(psf.get_atype(), new_coord_df.to_numpy()))
    for i, (st, end) in enumerate(zip(ibase[:-1], ibase[1:])):
        cur_res = chain.residues[i]
        for atom_name, coordinate in atom_coords[st:end]:
            if atom_name in cur_res:
                cur_res[atom_name].coord = coordinate
    print(f'[crimm] Synchronized: {chain}')

def get_charmm_coord_dict(selected_atoms, include_resname = True):
    """Get a dictionary of coordinates of selected atoms from CHARMM.
    The dictionary is organized by SEGID, and then by residue sequence and atom name.
    """
    # atom_idx from CHARMM is zero-indexed
    atom_idx = np.array(selected_atoms.get_atom_indexes()) 
    pos = pcm.coor.get_positions().to_numpy()
    atom_pos = pos[atom_idx]
    resseq = []
    for id_str in selected_atoms.get_res_ids():
        if not id_str.isdigit():
            ## CHARMM's residue ID can be in the format of "1A", "1B", etc.
            id_str = ''.join([c for c in id_str if c.isdigit()])
        resseq.append(int(id_str))
    resnames = selected_atoms.get_res_names()
    a_types = selected_atoms.get_atom_types()
    segids = selected_atoms.get_seg_ids()
    
    coords_dict = {}
    for segid, resseq, resname, a_name, coords in zip(
        segids, resseq, resnames, a_types, atom_pos
    ):
        if segid not in coords_dict:
            coords_dict[segid] = {}
        if not include_resname:
            coords_dict[segid][(resseq, a_name)] = coords
        else:
            coords_dict[segid][(resname, resseq, a_name)] = coords
    return coords_dict

def get_missing_water_h_dict(model):
    missing_hydrogen_dict = {}
    for water_chain in model.solvent:
        for water in water_chain:
            if len(water.missing_hydrogens) == 0:
                continue
            if water.segid not in missing_hydrogen_dict:
                missing_hydrogen_dict[water.segid] = {}
            for atom_name, missing_h in water.missing_hydrogens.items():
                missing_h_key = (water.id[1], atom_name)
                missing_hydrogen_dict[water.segid][missing_h_key] = missing_h
    return missing_hydrogen_dict

def _build_water_with_dicts(missing_water_h_dict, h_coords_dict):
    for h_key, missing_h in missing_water_h_dict.items():
        water_res = missing_h.parent
        if h_key not in h_coords_dict:
            raise KeyError(
                f'Corresponding hydrogen coordinates for {missing_h} in {water_res} not found in CHARMM'
            )
        missing_h.coord = h_coords_dict[h_key]
        h_name = h_key[-1]
        water_res.missing_hydrogens.pop(h_name)
        water_res.add(missing_h)

def create_water_hs_from_charmm(model):
    """Create missing hydrogen atoms in water residues from CHARMM."""
    missing_water_h_dicts = get_missing_water_h_dict(model)
    for segid, missing_h_dict in missing_water_h_dicts.items():
        # build water hydrogen atoms in CHARMM
        pcm.lingo.charmm_script(
            f'hbuild sele SEGI {segid} .and. .not. TYPE O* end'
        )
        charmm_water_hs = (
            pcm.SelectAtoms().by_seg_id(segid) & 
            pcm.SelectAtoms().all_hydrogen_atoms()
        )
        h_coords_dict = get_charmm_coord_dict(
            charmm_water_hs, include_resname=False
        )
        _build_water_with_dicts(missing_h_dict, h_coords_dict[segid])

def fetch_coords_from_charmm(entity):
    """Fetch coordinates from CHARMM to the entity."""
    res_list = unfold_entities(entity, 'R')
    all_charmm_atoms = pcm.SelectAtoms().all_atoms()
    charmm_coord_dict = get_charmm_coord_dict(all_charmm_atoms)
    for residue in res_list:
        atoms = list(residue.get_atoms())
        # Update lone pairs coordinates for heterogens too
        if isinstance(residue, Heterogen):
            atoms.extend(residue.lone_pairs)
        resname = residue.resname
        if resname == 'HIS':
            # use histidine's CHARMM name
            resname = residue.topo_definition.resname
        if resname == 'HOH':
            resname = 'TIP3'
        segid = residue.segid
        if segid == ' ':
            raise ValueError(f'No SEGID assigned to {residue}')
        resseq = residue.id[1]
        for atom in atoms:
            atom_name = atom.name
            atom_key = (resname, resseq, atom_name)
            if segid not in charmm_coord_dict:
                raise KeyError(
                    f'{atom} in {residue} with SEGID {segid} not found in CHARMM'
                )
            if atom_key not in charmm_coord_dict[segid]:
                raise KeyError(
                    f'{atom} in {residue} not found in CHARMM SEGMENT {segid}'
                    f' with RESNAME {resname} RESID {resseq} ATOM NAME {atom_name}'
                )
            atom.coord = charmm_coord_dict[segid][atom_key]

def minimize(constrained_atoms='CA', sd_nstep=1000, abnr_nstep=500):
    """Simple minimization with harmonic constraints on specified atoms.

    This is a convenience function that runs steepest-descent followed by
    adopted basis Newton-Raphson minimization with CA atoms constrained.

    Parameters
    ----------
    constrained_atoms : str, default 'CA'
        Atom type to constrain with harmonic potential (e.g., 'CA' for alpha carbons)
    sd_nstep : int, default 1000
        Number of steepest-descent minimization steps (0 to skip)
    abnr_nstep : int, default 500
        Number of adopted basis Newton-Raphson minimization steps (0 to skip)
    """
    cons_harm.setup_absolute(
        selection=pcm.SelectAtoms(atom_type=constrained_atoms),
        force_constant=50
    )
    if int(sd_nstep) > 0:
        _minimize.run_sd(nstep=int(sd_nstep))
    else:
        print('Steepest-descend minimization not performed')
    if int(abnr_nstep) > 0:
        _minimize.run_abnr(nstep=int(abnr_nstep), tolenr=1e-3, tolgrd=1e-3)
    else:
        print('Adopted Basis Newton-Raphson minimization not performed')
    cons_harm.turn_off()

def sd_minimize(
    nstep, non_bonded_script, tolenr=1e-3, tolgrd=1e-3, 
    cons_harm_selection=None, harm_force_const=20, cons_fix_selection=None
):
    """Perform steepest-descent minimization in CHARMM."""
    # Implement the non-bonded parameters by "running" them.
    non_bonded_script.run()
    # equivalent to: 
    # cons harm force 20 select type ca end
    has_cons_harm = False
    has_cons_fix = False
    if cons_harm_selection is not None:
        if cons_harm_selection.get_n_selected() == 0:
            warnings.warn(
                "Atom selection resulted zero atoms for CONS HARM! Skip CONS HARM setup"
            )
        else:
            status = cons_harm.setup_absolute(
                selection=cons_harm_selection,
                force_const=harm_force_const
            )
            has_cons_harm = not status
            # The status would return False if success
            warnings.warn(f"Absolute harmonic restraints setup success: {has_cons_harm}")
    elif cons_fix_selection is not None:
        if cons_fix_selection.get_n_selected() == 0:
            warnings.warn(
                "Atom selection resulted zero atoms for CONS FIX! Skip CONS FIX setup"
            )
        else:
            has_cons_fix = cons_fix.setup(cons_fix_selection)
            warnings.warn(f"Atom fix constraint setup success: {has_cons_fix}")
    # equivalent CHARMM scripting command: 
    # minimize abnr nstep 1000 tole 1e-3 tolgr 1e-3
    _minimize.run_sd(nstep=nstep, tolenr=tolenr, tolgrd=tolgrd)
    if has_cons_harm:
        cons_harm.turn_off()
    if has_cons_fix:
        cons_fix.turn_off()
    # equivalent CHARMM scripting command: energy
    ener_df = energy.get_energy()
    return ener_df.iloc[0].to_dict()