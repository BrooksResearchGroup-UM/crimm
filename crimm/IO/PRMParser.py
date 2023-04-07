from collections import namedtuple
from crimm.IO.RTFParser import comment_parser, skip_line

bond_par = namedtuple('bond_param',['kb', 'b0'])
angle_par = namedtuple('angle_param',['ktheta', 'theta0'])
ub_par = namedtuple('ub_param',['kub', 's0']) # Urey-Bradley
dihe_par = namedtuple('dihedral_param',['kchi', 'n', 'delta'])
impr_par = namedtuple('improper_param',['kpsi', 'psi0'])
cmap_par = namedtuple('cmap_param',['degree', 'values'])
nonbond_par = namedtuple('nonbond_param', ['epsilon', 'rmin_half'])
nonbond14_par = namedtuple('nonbond14_param', ['epsilon', 'rmin_half'])
nbfix_par = namedtuple('nbfix_param', ['emin','rmin'])

def categorize_lines(lines):
    line_dict = {
        'mass': [], 'bonds': [], 'angles': [], 'dihedrals': [],
        'improper': [], 'cmap': [], 'nonbonded': [], 'nbfix':[]
    }
    is_continued = False
    for line in lines:
        l = line.strip().upper()
        if skip_line(l):
            continue
        if l.endswith('-'):
            if is_continued:
                is_continued += l.rstrip('-')
            else:
                is_continued = l.rstrip('-')
            continue
        if is_continued:
            l = is_continued
            is_continued = False
        if l.startswith('ATOMS'):
            cur_section = line_dict['mass']
        elif l.startswith('BONDS'):
            cur_section = line_dict['bonds']
        elif l.startswith('ANGLES'):
            cur_section = line_dict['angles']
        elif l.startswith('DIHEDRALS'):
            cur_section = line_dict['dihedrals']
        elif l.startswith('IMPROPER'):
            cur_section = line_dict['improper']
        elif l.startswith('CMAP'):
            cur_section = line_dict['cmap']
        elif l.startswith('NONBONDED'):
            cur_section = line_dict['nonbonded']
        elif l.startswith('NBFIX'):
            cur_section = line_dict['nbfix']
        elif l.startswith('HBOND'):
            pass
        elif l.startswith('END'):
            break
        else:
            cur_section.append(l)
    return line_dict

def bond_parser(line):
    field_str, desc = comment_parser(line)
    # fields: [atom_type1, atom_type2, Kb: kcal/mole/A**2, b0: A]
    atom1, atom2, kb, b0 = field_str.split()
    return (atom1, atom2), bond_par(float(kb), float(b0))

def angle_parser(line):
    # V(angle) = Ktheta(Theta - Theta0)**2
    # V(Urey-Bradley) = Kub(S - S0)**2
    # fields : [
    #     atom_type1, atom_type2, atom_type3,
    #     Ktheta: kcal/mole/rad**2, Theta0: degrees
    #     Kub: kcal/mole/A**2 (Urey-Bradley), S0: A
    # ]
    field_str, desc = comment_parser(line)
    fields = field_str.split()
    if len(fields) == 7:
        atom1, atom2, atom3, ktheta, theta0, kub, s0 = fields
        return (
            (atom1, atom2, atom3), 
            angle_par(float(ktheta), float(theta0)), 
            ub_par(float(kub), float(s0))
        )
    atom1, atom2, atom3, ktheta, theta0 = fields
    return (
        (atom1, atom2, atom3), 
        angle_par(float(ktheta), float(theta0)), 
        None
    )

def dihedral_parser(line):
    # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
    # fields: [
    #     atom1, atom2, atom3, atom4, 
    #     Kchi: kcal/mole, n: multiplicity, delta: degrees
    # ]
    field_str, desc = comment_parser(line)
    a1, a2, a3, a4, kchi, n, delta = field_str.split()
    return (a1,a2,a3,a4), dihe_par(float(kchi), int(n), float(delta))

def improper_parser(line):
    # V(improper) = Kpsi(psi - psi0)**2
    # fields: [
    #     atom1, atom2, atom3, atom4, 
    #     Kpsi: kcal/mole/rad**2, ignored, psi0: degrees
    # ]
    field_str, desc = comment_parser(line)
    a1, a2, a3, a4, kpsi, _, psi0 = field_str.split()
    return (a1,a2,a3,a4), impr_par(float(kpsi), float(psi0))

def cmap_key_parser(line):
    # fields: [8 atoms, n_cmap_entries]
    fields = line.split()
    return tuple(fields[:8]), int(fields[-1])

def cmap_field_parser(lines, n):
    entries = []
    for line in lines:
        entries.extend([float(entry) for entry in line.split()])
    assert(len(entries) == n**2)
    return entries

def nonbonded_parser(line):
    # V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
    # fields: [
    #     atom, ignored
    #     epsilon: kcal/mole, 
    #     Eps,i,j = sqrt(eps,i * eps,j)
    #     Rmin/2 (Rmin,i,j = Rmin/2,i + Rmin/2,j): A, 
    #     ignored, eps,1-4, Rmin/2,1-4
    # ]
    field_str, desc = comment_parser(line)
    field = field_str.split()
    if len(field) == 7:
        atom, _, eps, rmin_2, _, eps_14, rmin_2_14 = field
        return (
            atom, 
            nonbond_par(float(eps), float(rmin_2)), 
            nonbond14_par(float(eps_14), float(rmin_2_14))
        )
    atom, _, eps, rmin_2 = field
    return atom, nonbond_par(float(eps), float(rmin_2)), None

def nbfix_parser(line):
    # fields:[atom1, atom2, Emin: kcal/mol, Rmin: A]
    field_str, desc = comment_parser(line)
    atom1, atom2, emin, rmin = field_str.split()
    return (atom1, atom2), nbfix_par(float(emin), float(rmin))

def simple_collector(param_line_dict, parser):
    param_dict = {}
    for l in param_line_dict:
        key, val = parser(l)
        if key in param_dict:
            continue
        param_dict[key] = val
    return param_dict

def dihedral_collector(param_line_dict, parser):
    param_dict = {}
    for l in param_line_dict:
        key, val = parser(l)
        if key not in param_dict:
            param_dict[key] = []
        param_dict[key].append(val)
    for value_list in param_dict.values():
        value_list.sort(key=lambda x: x.n)
    return param_dict

def extended_collector(param_line_dict, parser):
    param_dict = {}
    extended_param_dict = {}
    for l in param_line_dict:
        key, val1, val2 = parser(l)
        param_dict[key] = val1
        if val2 is not None:
            extended_param_dict[key] = val2
    return param_dict, extended_param_dict

def parse_line_dict(line_dict):

    bond_dict = simple_collector(line_dict['bonds'], bond_parser)
    angle_dict, urey_bradley_dict = extended_collector(
        line_dict['angles'], angle_parser
    )
    dihe_dict = dihedral_collector(line_dict['dihedrals'], dihedral_parser)
    impr_dict = simple_collector(line_dict['improper'], improper_parser)
    
    cmap_lines = {}
    for l in line_dict['cmap']:
        if not (l[0].isdigit() or l[0] == '-'):
            key, n = cmap_key_parser(l)
            cmap_lines[key] = {'n':n, 'entries':[]}
            cur_cmap_lines = cmap_lines[key]['entries']
        else:
            cur_cmap_lines.append(l)

    cmap_dict = {}
    for k, v in cmap_lines.items():
        n, entry_lines = v.values()
        deg_range = range(-180, 180, 360//n)
        id_range = range(0,n**2+1,n)
        entries = cmap_field_parser(entry_lines, n) 
        cmap_dict[k] = []
        for deg, st, end in zip(deg_range, id_range[:-1], id_range[1:]):
            cmap_dict[k].append(cmap_par(deg, entries[st:end]))

    nonbond_dict, nonbond14_dict = extended_collector(
        line_dict['nonbonded'], nonbonded_parser
    )

    nbfix_dict = simple_collector(line_dict['nbfix'], nbfix_parser)
        
    return {
        'bonds': bond_dict,
        'angles': angle_dict,
        'urey_bradley': urey_bradley_dict,
        'dihedrals': dihe_dict,
        'improper': impr_dict,
        'cmap': cmap_dict,
        'nonbonded': nonbond_dict,
        'nonbonded14': nonbond14_dict,
        'nbfix': nbfix_dict
    }