import os
import pycharmm as chm
from pycharmm import read, write
from pycharmm import generate
from pycharmm import ic, cons_harm
from pycharmm import minimize

def check_list_validity(file_list):
    if not isinstance(file_list, list) or len(file_list) == 0:
        raise TypeError('a list of file paths is required')
    for file in file_list:
        if not os.path.exists(file):
            raise ValueError(f'invalid file path: {file}')

def load_params(rtf_list, prm_list):
    # verify the conformance of input datatype
    check_list_validity(rtf_list)
    check_list_validity(prm_list)
    
    for i, rtf in enumerate(rtf_list):
        # any file path after the first one will activate append mode
        read.rtf(rtf, append=bool(i))

    for i, prm in enumerate(prm_list):
        read.prm(prm, append=bool(i), flex=True)

def load_structure(
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

def double_minimize(
        constrained_atoms='CA', 
        sd_nstep=1000,
        abnr_nstep=500,
    ):

    cons_harm.setup_absolute(
        selection=chm.SelectAtoms(atom_type=constrained_atoms),
        force_constant=50
        )
    if int(sd_nstep) > 0:
        minimize.run_sd(nstep=int(sd_nstep))
    else:
        print('Steepest-descend minimization not performed')
    if int(abnr_nstep) > 0:
        minimize.run_abnr(nstep=int(abnr_nstep), tolenr=1e-3, tolgrd=1e-3)
    else:
        print('Adopted Basis Newton-Raphson minimization not performed')
    cons_harm.turn_off()