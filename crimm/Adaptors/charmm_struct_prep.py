from crimm.Fetchers import fetch_rcsb
from crimm.Modeller import ParameterLoader, TopologyLoader
from crimm.Modeller.LoopBuilder import ChainLoopBuilder
from crimm.Modeller.TopoFixer import fix_chain
from crimm.StructEntities import Model
import crimm.Adaptors.pyCHARMMAdaptors as pcm_interface
from crimm.Adaptors.PropKaAdaptors import PropKaProtonator

from pycharmm.psf import delete_atoms as pcm_del_atoms
from pycharmm.psf import get_natom as pcm_get_natom
from pycharmm.generate import patch as pcm_patch
from pycharmm.settings import set_verbosity as pcm_set_verbosity

def minimize_chain(chain, sd_nstep, abnr_nstep):
    """Minimize crimm protein chain in CHARMM"""
    # load into CHARMM to minimize the structure
    if pcm_get_natom() > 0:
        pcm_del_atoms()
    pcm_interface.load_chain(chain)
    pcm_interface.minimize(sd_nstep=sd_nstep, abnr_nstep=abnr_nstep)
    # Uodate the coordinate in crimm structure
    pcm_interface.sync_coords(chain)

def correct_prot_first_patch(chain, default):
    """Correct the first patch name for PRO and GLY"""
    # PRO and GLY need special treatment when patched at the N-terminus 
    first_resname = chain.residues[0].resname
    if first_resname == 'PRO':
        first_patch = 'PROP'
    elif first_resname == 'GLY':
        first_patch = 'GLYP'
    else:
        first_patch = default
    return first_patch

def get_param_loaders():
    """get crimm rtf and prm loader for protein and nucleic acid chains
    also load them into CHARMM"""
    rtf_loader = {
        'prot': TopologyLoader('protein'),
        'na': TopologyLoader('nucleic')
    }
    param_loader = {
        'prot': ParameterLoader('protein'),
        'na': ParameterLoader('nucleic')
    }

    # fill the missing ic table values in the respective rtf
    for i, (chain_type, cur_rtf) in enumerate(rtf_loader.items()):
        cur_param = param_loader[chain_type]
        cur_param.fill_ic(cur_rtf)
        # load the respective files into CHARMM as well
        prev_level = pcm_set_verbosity(0)
        pcm_interface.load_topology(cur_rtf, append=bool(i))
        pcm_interface.load_parameters(cur_param, append=bool(i))
        pcm_set_verbosity(prev_level)
    return rtf_loader, param_loader

def protonate_and_patch(model, pH, rtf_loader, param_loader):
    """protonate all protein chains in a model and also load in charmm and
    patch the residues"""
    protonator = PropKaProtonator(
        rtf_loader, param_loader, pH = pH
    )
    protonator.load_model(model)
    protonator.apply_patches()

    if pcm_get_natom() > 0:
        pcm_del_atoms()
    for chain in model:
        if chain.id in protonator.patches and len(protonator.patches[chain.id]) > 0:
            built_atoms = fix_chain(chain)
        pcm_interface.load_chain(chain)

    for chain_id, patch_dict in protonator.patches.items():
        for resid, patch_name in patch_dict.items():
            pcm_patch(patch_name, f'PRO{chain_id} {resid}')

def load_pdbid_in_charmm(
    pdb_id,
    pH = 7.4,
    prot_first_patch = 'ACE',
    prot_last_patch = 'CT3',
    na_first_patch = '5TER',
    na_last_patch = '3PHO',
    sd_nstep = 300,
    abnr_nstep = 0,
    charmm_verbosity_level = 0,
):

    structure = fetch_rcsb(
        pdb_id,
        include_solvent=False,
        # any existing hydrogen will be removed and rebuilt later
        include_hydrogens=False,
        first_model_only=True
    )

    prot_chains = {}
    na_chains = {} 
    # get the first model's id
    model_id = structure.models[0].id
    # create a new empty model to store chains of interests
    new_model = Model(model_id)
    rtf_loader, param_loader = get_param_loaders()
    
    for chain in structure.models[0].chains:
        if chain.chain_type == 'Polypeptide(L)':
            prot_chains[chain.id] = chain
        elif chain.chain_type  == 'Polyribonucleotide':
            na_chains[chain.id] = chain

    for chain_id, chain in prot_chains.items():
        need_minimization = False
        # Missing loop in the chain
        if not chain.is_continuous():
            loop_builder = ChainLoopBuilder(chain)
            # Coordinates of the missing residues will be copied from
            # Alphafold structures
            # only build the loop not the termini
            loop_builder.build_from_alphafold(include_terminal = False)
            chain = loop_builder.get_chain()
            prot_chains[chain_id] = chain
            need_minimization = True
        prot_first_patch = correct_prot_first_patch(chain, default = prot_first_patch)
        rtf_loader['prot'].generate_chain_topology(
            chain,
            first_patch=prot_first_patch, 
            last_patch=prot_last_patch,
            # Coerce any modified residue to canonical residue that it is based on
            coerce=True
        )
        param_loader['prot'].fill_ic(rtf_loader['prot'])
        param_loader['prot'].apply(chain.topo_elements)
        fix_chain(chain)
        if need_minimization:
            # load into CHARMM to minimize the structure
            prev_level = pcm_set_verbosity(charmm_verbosity_level)
            minimize_chain(chain, sd_nstep, abnr_nstep)
            pcm_set_verbosity(prev_level)
        new_model.add(chain)

    for chain_id, chain in na_chains.items():
        # Missing loop is very unlikely in nucleotide chains on PDB
        # but if it exsits, an error will be raise
        if not chain.is_continuous():
            raise ValueError(
                f'Nucleotide chain {chain.id} is not continuous, '
                'topology cannot be generated.'
            )
        rtf_loader['na'].generate_chain_topology(
            chain, 
            first_patch=na_first_patch,
            last_patch=na_last_patch,
            coerce=True
        )
        param_loader['na'].fill_ic(rtf_loader['na'])
        param_loader['na'].apply(chain.topo_elements)
        fix_chain(chain)
        new_model.add(chain)
    # copy the connection record to the new model
    new_model.set_connect(structure.models[0].connect_dict)
    # replace the model with the new model in structure
    structure.detach_child(model_id)
    structure.add(new_model)
    protonate_and_patch(
        new_model, pH, rtf_loader['prot'], param_loader['prot']
    )
    # patch disulfide bonds if exists
    if 'disulf' in structure.models[0].connect_dict:
        for res1, res2 in structure.models[0].connect_dict['disulf']:
            seg1, seg2 = res1['chain'], res2['chain']
            seq1, seq2 = res1['resseq'], res2['resseq']
            patch_arg = f'PRO{seg1} {seq1} PRO{seg2} {seq2}'
            pcm_patch('DISU', patch_arg)

    return structure

if __name__ == '__main__':
    import warnings
    import argparse
    from pycharmm import write
    parser = argparse.ArgumentParser(
        prog="CHARMM Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
'''
  CHARMM Structure Loader from PDB ID
    Load a structure fetched from rcsb with PDB ID. Any missing loop will be constructed
    from the coordinates of the Alphafold structure. Protonation is done with PropKa model.
    Only protein and RNA chain will be processed, and only protein chain will be protonated
    with the specified pH value.
'''
    )
    parser.add_argument(
        'pdbid', metavar='P', type=str, help='PDB ID for the structure'
    )
    parser.add_argument(
        '-o','--outpath', type=str, default="./",
        help='Directory path for pdb and psf file outputs. '
        'Default to current directory'
    )
    parser.add_argument(
        '-p', '--ph', type=float, default=7.4,
        help='pH value for the protonation state for protein'
        'Default to pH=7.4.'
    )
    parser.add_argument(
        '-m', '--mini', type=int, default=300,
        help='Minimization steps to take after building any missing loop. '
        'Default to 300 steps of steepest descent.'
    )
    parser.add_argument(
        '--report', action='store_true',
        help='If the report flag is present, the crimm structure information '
        'will also be displayed.'
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    # test_ids = ['5IEV', '1A8I', '7ZAP']
    structure = load_pdbid_in_charmm(
        args.pdbid,
        pH = args.ph,
        prot_first_patch = 'ACE',
        prot_last_patch = 'CT3',
        na_first_patch = '5TER',
        na_last_patch = '3PHO',
        sd_nstep = args.mini,
        abnr_nstep = 0,
        charmm_verbosity_level = 0,
    )
    write.coor_pdb(f'{args.pdbid}.pdb')
    write.psf_card(f'{args.pdbid}.psf')
    
    if args.report:
        print(structure)
        total_atoms = len(list(structure.get_atoms()))
        total_segments = len(structure.models[0])
        total_residues = 0
        total_bonds = 0
        total_angles = 0
        total_dihe = 0
        total_impr = 0
        for chain in structure.models[0].chains:
            print(chain.topo_elements)
            total_residues += len(chain)
            total_bonds += len(chain.topo_elements.bonds)
            total_angles += len(chain.topo_elements.angles)
            total_dihe += len(chain.topo_elements.dihedrals)
            total_impr += len(chain.topo_elements.impropers)
        print(
            'Report from crimm (Total)',
            f'segments={total_segments}, residues={total_residues}, atoms={total_atoms}, '
            f'bonds={total_bonds}, angles={total_angles}, dihedrals={total_dihe}, '
            f'impropers={total_impr}'
        )
    