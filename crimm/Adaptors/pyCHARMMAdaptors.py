import tempfile
import pycharmm as chm
from pycharmm import read, psf, coor
from pycharmm import generate
import pycharmm.settings as pcm_settings
from pycharmm import ic, cons_harm
from pycharmm import minimize as _minimize
from pycharmm.psf import get_natom, delete_atoms
from crimm.IO import get_pdb_str
from pathlib import Path

nucleic_letters_1to3 = {
    'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'T': 'THY', 'U': 'URA',
}

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
            tf.write(f'*{topo_type.upper()} RTF Loaded from crimm\n')
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
            tf.write(f'*{param_type.upper()} PRM Loaded from crimm\n')
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
                tf.write('*CGENFF RTF Loaded from crimm\n')
                tf.write(f.read())
                tf.flush() # has to flush first for long files!
                read.rtf(tf.name, append = True)

        pcm_settings.set_bomb_level(-1)
        with open(prm_abs_path, 'r', encoding='utf-8') as f:
            with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
                tf.write('*CGENFF PRM Loaded from crimm\n')
                tf.write(f.read())
                tf.flush() # has to flush first for long files!
                read.prm(tf.name, append=True, flex=True)
        pcm_settings.set_bomb_level(0)

        ligandrtf_blocks = topo_generator.cgenff_loader._raw_data_blocks
        for data_block in ligandrtf_blocks:
            with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
                tf.write(f'* CGENFF LIGAND RTF Loaded from crimm\n')
                tf.write(data_block)
                tf.flush() # has to flush first for long files!
                read.stream(tf.name)
        
    # load water_ions.str at the end
    if load_water_ions:
        abs_path = Path(__file__).resolve().parent.parent
        abs_path = abs_path / "Data/toppar/water_ions.str"
        with open(abs_path, 'r', encoding='utf-8') as f:
            with tempfile.NamedTemporaryFile('w', encoding = "utf-8") as tf:
                tf.write('*WATER ION TOPPAR Loaded from crimm\n')
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
    return segid

def load_ligands(ligand_chains):
    segids = []
    all_ligands = [res for chain in ligand_chains for res in chain] 
    for i, lig_res in enumerate(all_ligands, start=1):
        segid = f'LG{i:02d}'
        lig_res.segid = segid
        print(f"Loading ligand {lig_res.resname} SEG: {segid}")
        with tempfile.NamedTemporaryFile('w') as tf:
            tf.write(get_pdb_str(lig_res, use_charmm_format=True))
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

def load_water(water_chains):
    # Currently only supports TIP3 water model
    segids = []
    for i, chain in enumerate(water_chains):
        chain = chain.copy()
        segid = f'WT{i:02d}'
        print(f"Loading water chain {segid}")
        for res in chain:
            res.segid = segid
            res.resname = res.topo_definition.resname
        with tempfile.NamedTemporaryFile('w') as tf:
            tf.write(get_pdb_str(chain, use_charmm_format=True))
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
        # we need to copy the chain here, since we might modify the resname and segid
        chain = chain.copy()
        segid = f'IO{i:02d}'
        print(f"Loading ion chain {segid}")
        for res in chain:
            res.segid = segid
            res.resname = res.resname.upper()
        with tempfile.NamedTemporaryFile('w') as tf:
            tf.write(get_pdb_str(chain, use_charmm_format=True))
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
    m_chain = chain.copy()
    if chain.chain_type == 'Polypeptide(L)':
        for res in m_chain:
            res.segid = segid
            if res.resname == 'HIS':
                # we uses HSD for default HIS protonation state here
                res.resname = res.topo_definition.resname
    elif chain.chain_type == 'Polyribonucleotide':
        for res in m_chain:
            res.segid = segid
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
        chm.lib.charmm.ic_print()
    if hbuild:
        chm.lingo.charmm_script("hbuild sele type H* end")

def minimize(
        constrained_atoms='CA', 
        sd_nstep=1000,
        abnr_nstep=500,
    ):

    cons_harm.setup_absolute(
        selection=chm.SelectAtoms(atom_type=constrained_atoms),
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
    
def ok_to_sync(chain):
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
    if not ok_to_sync(chain):
        print("[Abort] Possible residue sequences mismatch!")
        return
    ibase = list(psf.get_ibase())
    new_coord_df = coor.get_positions()
    atom_coords = list(zip(psf.get_atype(), new_coord_df.to_numpy()))
    for i, (st, end) in enumerate(zip(ibase[:-1], ibase[1:])):
        cur_res = chain.residues[i]
        for atom_name, coordinate in atom_coords[st:end]:
            if atom_name in cur_res:
                cur_res[atom_name].coord = coordinate
    print(f'Synchronized: {chain}')
