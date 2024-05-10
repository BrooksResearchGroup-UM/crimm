NUCLEOSIDE_PHOS = ( # Nucleoside phosphates and phosphonates
    'ATP', 'ADP', 'AMP', 'ANP', 'ACP', 
    'CTP', 'CDP', 'CMP',
    'GCP', 'GTP', 'GDP', 'GNP',
    'UTP', 'UDP', 'UMP', 'UNP', 'UCP', 
    'TMP', 'TTP',
    'IMP', 'IDP'
)
# GlYCOSYLATION = ('NAG', 'NDG', 'NGA')
# MONOSACCGARIDES = (
#     'GLC', 'GAL', 'FRU', '3MK', '4GL', '4N2', 'AFD',
#     'ALL', 'ALT', 'ARA', 'BGC', 'BMA', 'BXY', 'CEL',
#     'DGL', 'FUC', 'GIV', 'GL0', 'GLA', 'GUP', 'BDF',
#     'GXL', 'MAN', 'SDY', 'SHD', 'WOO', 'Z0F', 'Z2D',
#     'Z6H', 'Z8H', 'Z8T', 'ZCD', 'ZEE', 
# )
glucopyranose = 'C(C1C(C(C(C(O1)O)O)O)O)O'
amino_deoxy_glucopyranose = 'NC1C(C(C(OC1O)CO)O)O'
fructofuranose = 'C(C1C(C(C(O1)(CO)O)O)O)O'
fructopyranose = 'C1C(C(C(C(O1)(CO)O)O)O)O'

ALKALI_METALS = ('Li','Na','K','Rb','Cs','Fr')
ALKALI_EARTH_METALS = ('Be','Mg','Ca','Sr','Ba','Ra')
TRANSITION_METALS = (
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs'
)
ALL_METALS = tuple((*ALKALI_METALS, *ALKALI_EARTH_METALS, *TRANSITION_METALS))
