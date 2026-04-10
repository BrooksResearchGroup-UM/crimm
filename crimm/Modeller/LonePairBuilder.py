"""Build lone pair coordinates from host atom positions.

Implements CHARMM's LONEPRC subroutine (lonepair.F90:617-820).
Supports all 4 LP geometry types: COLINEAR, RELATIVE, BISECTOR, CENTER.
"""
import warnings

import numpy as np
from numpy import linalg as LA


def build_colinear_coord(j_coord, k_coord, distance, scale=0.0):
    """Place LP along the J->K bond axis, on the far side of J from K.

    From LONEPRC (N=2 case, lonepair.F90:738-757):
        r_JK = J - K
        dr = scale + distance / |r_JK|
        LP = J + dr * r_JK

    Parameters
    ----------
    j_coord : ndarray
        Host atom position (the atom LP is attached to)
    k_coord : ndarray
        Bonded-to atom position (defines the bond axis)
    distance : float
        LP distance parameter (DIST from RTF)
    scale : float
        Scale factor (SCAL from RTF, default 0.0)
    """
    r_jk = j_coord - k_coord
    norm = LA.norm(r_jk)
    if norm < 1e-5:
        raise ValueError("COLINEAR host atoms at same position")
    dr = scale + distance / norm
    return j_coord + dr * r_jk


def build_relative_coord(j_coord, k_coord, l_coord, distance, angle, dihedral):
    """Place LP via internal coordinate (CARTCV) from 3 host atoms.

    From LONEPRC (N=3, V1>0 case, lonepair.F90:780-782):
        CALL CARTCV(X,Y,Z, L, K, J, I, V1, V2, V3, OK)

    The LP is built as atom I from reference atoms L, K, J using
    bond=distance, angle=angle, dihedral=dihedral.
    """
    from crimm.Modeller.TopoFixer import get_coord_from_dihedral_ic
    return get_coord_from_dihedral_ic(
        l_coord, k_coord, j_coord, dihedral, angle, distance
    )


def build_bisector_coord(j_coord, k_coord, l_coord, distance, angle, dihedral):
    """Place LP at angle bisector of K-J-L, then via CARTCV.

    From LONEPRC (N=3, V1<0 case, lonepair.F90:783-789):
        mid = (K + L) / 2
        CALL CARTCV(X,Y,Z, L, mid, J, I, |V1|, V2, V3, OK)
    """
    from crimm.Modeller.TopoFixer import get_coord_from_dihedral_ic
    mid = (k_coord + l_coord) / 2.0
    return get_coord_from_dihedral_ic(
        l_coord, mid, j_coord, dihedral, angle, distance
    )


def build_center_coord(host_coords):
    """Place LP at center of geometry of host atoms.

    From LONEPRC (QCENT case, lonepair.F90:792-809):
        LP = mean(host positions)
    """
    return np.mean(host_coords, axis=0)


def build_lonepair_coords(entity):
    """Position all lone pair atoms in an entity from their host coordinates.

    Iterates over all residues, finds LP atoms with lonepair_info,
    and computes their coordinates based on the geometry type.

    Parameters
    ----------
    entity : Model, Chain, OrganizedModel, or Residue
        Entity containing residues with lone_pair_dict populated.

    Returns
    -------
    int
        Number of LP atoms positioned.
    """
    count = 0
    if hasattr(entity, "get_residues"):
        residues = entity.get_residues()
    elif hasattr(entity, "lone_pair_dict"):
        residues = (entity,)
    else:
        raise ValueError(
            "Entity must provide get_residues() or lone_pair_dict for LP building."
        )

    for residue in residues:
        if not residue.lone_pair_dict:
            continue
        for lp_name, lp_atom in residue.lone_pair_dict.items():
            lp_def = lp_atom.topo_definition
            if lp_def is None or lp_def.lonepair_info is None:
                continue
            info = lp_def.lonepair_info
            lp_type = info['type']
            hosts = info['host_atoms']

            # Validate all host atoms exist with coordinates
            host_coords = []
            valid = True
            for h in hosts:
                if h not in residue:
                    warnings.warn(
                        f"LP {lp_name}: host atom {h} not found in residue "
                        f"{residue.resname} {residue.id[1]}"
                    )
                    valid = False
                    break
                h_coord = residue[h].coord
                if h_coord is None:
                    warnings.warn(
                        f"LP {lp_name}: host atom {h} has no coordinates"
                    )
                    valid = False
                    break
                host_coords.append(h_coord)
            if not valid:
                continue

            try:
                if lp_type == 'COLI':
                    lp_atom.coord = build_colinear_coord(
                        host_coords[0], host_coords[1],
                        info['distance'], info.get('scale', 0.0)
                    )
                elif lp_type == 'RELA':
                    lp_atom.coord = build_relative_coord(
                        host_coords[0], host_coords[1], host_coords[2],
                        info['distance'], info['angle'], info['dihedral']
                    )
                elif lp_type == 'BISE':
                    lp_atom.coord = build_bisector_coord(
                        host_coords[0], host_coords[1], host_coords[2],
                        info['distance'], info['angle'], info['dihedral']
                    )
                elif lp_type == 'CENT':
                    lp_atom.coord = build_center_coord(host_coords)
                else:
                    warnings.warn(f"Unknown LP type '{lp_type}' for {lp_name}")
                    continue
                count += 1
            except (ValueError, LA.LinAlgError) as e:
                warnings.warn(f"Failed to build LP {lp_name}: {e}")
    return count
