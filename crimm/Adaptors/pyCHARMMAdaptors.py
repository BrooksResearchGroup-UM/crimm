import tempfile
import warnings
import numpy as np
import pycharmm as pcm
import warnings
import numpy as np
import pycharmm as pcm
from pycharmm import read, psf, coor
from pycharmm import generate
import pycharmm.settings as pcm_settings
import pycharmm.settings as pcm_settings
from pycharmm import ic, cons_harm, cons_fix
from pycharmm import energy, cons_fix
from pycharmm import energy
from pycharmm import minimize as _minimize
from pycharmm.generate import patch as charmm_patch
from pycharmm.generate import patch as charmm_patch
from pycharmm.psf import get_natom, delete_atoms
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Selection import unfold_entities
from crimm.IO import get_pdb_str
from crimm import StructEntities as Entities
from crimm.Data.components_dict import nucleic_letters_1to3
from pathlib import Path
from crimm import StructEntities as Entities
from crimm.Data.components_dict import nucleic_letters_1to3
from pathlib import Path

def empty_charmm():
    """If any atom exists in current CHARMM runtime, remove them."""
    if get_natom() > 0:
        delete_atoms()

def load_topology(topo_generator):
    """Load topology and parameter files from a TopoGenerator object."""
    load_water_ions = False
    load_cgenff = False
    if 'cgenff' in topo_generator.res_def_dict:
        load_cgenff = True
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
    if load_cgenff:
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

        ligandrtf_blocks = topo_generator.cgenff_loader.toppar_blocks
        for resname, data_block in ligandrtf_blocks.items():
            with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
                tf.write(f'* CGENFF TOPPAR for {resname} Loaded from crimm\n')
                tf.write(data_block)
                tf.flush() # has to flush first for long files!
                read.stream(tf.name)
        
    # load water_ions.str at the end
    if load_water_ions:
        abs_path = Path(__file__).resolve().parent.parent
        abs_path = abs_path / "Data/toppar/water_ions.str"
        with open(abs_path, 'r', encoding='utf-8') as f:
            with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
                tf.write('* WATER ION TOPPAR Loaded from crimm\n')
                for line in f.readlines():
                    tf.write(line)
                tf.flush()
                read.stream(tf.name)

def load_chain(chain, hbuild = False, report = False):
    if not chain.is_continuous():
        raise ValueError("Chain is not continuous! Fix the chain first!")
    ##TODO: change seg id based on chain type (use NUC for nucleic)
    segid = f'PRO{chain.id[0]}'
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

def load_ligands(ligand_chains, segids=None):
    """Load a list of ligand chains into CHARMM."""
    all_ligands = [res for chain in ligand_chains for res in chain]
    if segids is None:
        segids = [f'LG{i:02d}' for i in range(len(all_ligands))]
    elif len(segids) != len(all_ligands):
        raise ValueError("Number of segids must match number of ligands")

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

def load_water(water_chains, segids=None):
    # Currently only supports TIP3 water model
    if segids is None:
        segids = [f'WT{i:02d}' for i in range(len(water_chains))]
    elif len(segids) != len(water_chains):
        raise ValueError("Number of segids must match number of water chains")

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
        segids.append(segid)
    return segids

def load_ions(ion_chains):
    segids = []
    for i, chain in enumerate(ion_chains):
        segid = f'IO{i:02d}'
        print(f"[crimm] Loading ion chain {segid}")
        for res in chain:
            res.segid = segid
        # we need to copy the chain here, since we might modify atom_serial_number
        chain = chain.copy()
        chain.reset_atom_serial_numbers(reset_current_only=True)
        
            # res.resname = res.resname.upper()
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
