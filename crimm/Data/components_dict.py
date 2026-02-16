## for charmm pdb files
nucleic_letters_1to3 = {
    'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'T': 'THY', 'U': 'URA',
}

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

PDB_CHARMM_ION_NAMES = {
    "LI": "LIT",
    "NA": "SOD",
    "MG": "MG",
    "K": "POT",
    "CA": "CAL",
    "RB": "RUB",
    "CS": "CES",
    "BA": "BAR",
    "ZN": "ZN", # However, ZN2 is the resname used in CHARMM
    "CD": "CAD",
    "CL": "CLA",
}

CHARMM_PDB_ION_NAMES = {k:v for v,k in PDB_CHARMM_ION_NAMES.items()}

# Common co-solvent / crystallization additive residue names.
# Used as a local fallback to avoid network calls for known non-ligand heterogens.
COMMON_COSOLVENTS = frozenset({
    # Cryoprotectants / Precipitants
    'GOL',  # Glycerol
    'EDO',  # Ethylene glycol
    'PEG',  # Polyethylene glycol
    'MPD',  # 2-Methyl-2,4-pentanediol
    'PGE',  # Triethylene glycol
    'P6G',  # Hexaethylene glycol
    '1PE',  # Pentaethylene glycol
    '2PE',  # Nonaethylene glycol
    'PG4',  # Tetraethylene glycol
    'IPA',  # Isopropanol
    'EOH',  # Ethanol
    'MOH',  # Methanol
    'BU3',  # tert-Butanol
    'PDO',  # 1,3-Propanediol
    # Buffers
    'TRS',  # Tris buffer
    'MES',  # MES buffer
    'EPE',  # HEPES buffer
    'IMD',  # Imidazole
    'BCN',  # Bicine
    'CIT',  # Citrate
    'TAM',  # Tris (alternate)
    'BTB',  # Bis-tris propane
    'CAC',  # Cacodylate
    # Carboxylic acids
    'ACT',  # Acetate
    'FMT',  # Formate
    'ACY',  # Acetic acid
    'MLI',  # Malonate
    'TAR',  # D(+)-Tartaric acid
    'OXL',  # Oxalic acid
    'ACA',  # 6-Aminohexanoic acid
    # Other common additives
    'DMS',  # DMSO
    'BME',  # Beta-mercaptoethanol
    'SO4',  # Sulfate
    'PO4',  # Phosphate
    'SCN',  # Thiocyanate
    'NO3',  # Nitrate
    'NH4',  # Ammonium
    'BCT',  # Bicarbonate
})
