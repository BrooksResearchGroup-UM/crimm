#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void calc_pairwise_dist(
    double* grid_pos, double* coords, int N_COORDS, 
    int N_GRID_POINTS, double* dists
){
    double dx, dy, dz;
    for (int i = 0; i < N_GRID_POINTS; i++) {
        for (int j = 0; j < N_COORDS; j++) {
            // pairwise distances between the receptor atom coords 
            // and the grid points
            dx = grid_pos[i * 3] - coords[j * 3];
            dy = grid_pos[i * 3 + 1] - coords[j * 3 + 1];
            dz = grid_pos[i * 3 + 2] - coords[j * 3 + 2];
            dists[i * N_COORDS + j] = sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
}

double calc_point_elec_potential(
    double dist, double elec_const, double charge, double rc, double alpha,
    double elec_rep_max, double elec_attr_max
) {
    double cur_potential;
    // beyond cutoff
    if (dist > rc) {
        cur_potential = elec_const / (dist * dist);
    // within cutoff
    } else {
        double alpha_tmp = alpha * dist * dist;
        // repulsive
        if (charge > 0) {
            cur_potential = elec_rep_max - alpha_tmp;
        // attractive
        } else {
            cur_potential = elec_attr_max + alpha_tmp;
        }
        // zero charge is very unlikely and will 
        // turn out to be zero potential in the end
    }
    return cur_potential;
}

void gen_elec_grid(
    double* dists, double* charges, double cc_elec,
    double rad_dielec_const, double elec_rep_max, double elec_attr_max,
    int N_COORDS, int N_GRID_POINTS, double* electrostat_grid
) { 
    double* elec_consts = malloc(N_COORDS * sizeof(double));
    double* rc = malloc(N_COORDS * sizeof(double));
    double* alpha = malloc(N_COORDS * sizeof(double));

    double emax_tmp;
    double dist, cur_potential, cur_grid_val;

    for (int i = 0; i < N_COORDS; i++) {
        // calculate electrostatic constants
        elec_consts[i] = cc_elec * charges[i] / rad_dielec_const;
        if (charges[i] > 0) {
            // repulsive
            emax_tmp = elec_rep_max;
        } else {
            // attractive
            emax_tmp = elec_attr_max;
        }
        // cutoff distance
        rc[i] = sqrt(2.0 * fabs(elec_consts[i] / emax_tmp));
        alpha[i] = fabs(emax_tmp / (2.0 * rc[i] * rc[i]));
    }

    // calculate electrostatic grid values
    for (int i = 0; i < N_GRID_POINTS; i++) {
        cur_grid_val = 0.0;
        for (int j = 0; j < N_COORDS; j++) {
            dist = dists[i * N_COORDS + j];
            // calculate electrostatic potential
            cur_potential = calc_point_elec_potential(
                dist, elec_consts[j], charges[j], rc[j], alpha[j],
                elec_rep_max, elec_attr_max
            );
            cur_grid_val += cur_potential;
        }
        electrostat_grid[i] = cur_grid_val;
    }
    free(elec_consts);
    free(rc);
    free(alpha);
}

void gen_vdw_grid(
    double* dists, double* epsilons, double* vdw_rs, double probe_radius, 
    double vwd_softcore_max, int N_COORDS, int N_GRID_POINTS, double* vdw_grid
) {
    double* r_mins = malloc(N_COORDS * sizeof(double));
    double* eps_sqrt = malloc(N_COORDS * sizeof(double));
    double* rc_vdw = malloc(N_COORDS * sizeof(double));
    double* beta = malloc(N_COORDS * sizeof(double));

    double r_min, r_min_over_dist;
    double cur_eps_sqrt, vdwconst;
    double dist, cur_grid_val;

    // calculate vdw constants
    for (int i = 0; i < N_COORDS; i++) {
        r_min = vdw_rs[i] + probe_radius;
        r_mins[i] = r_min;
        cur_eps_sqrt = sqrt(fabs(epsilons[i]));
        eps_sqrt[i] = cur_eps_sqrt;
        vdwconst = 1.0 + sqrt(
            1.0 + 0.5 * fabs(vwd_softcore_max) / cur_eps_sqrt
        );
        // cutoff distance
        rc_vdw[i] = r_min * pow(vdwconst, -1.0 / 6.0);
        beta[i] = 24.0 * cur_eps_sqrt /
        vwd_softcore_max * (vdwconst * vdwconst - vdwconst);
    }

    for (int i = 0; i < N_GRID_POINTS; i++) {
        cur_grid_val = 0.0;
        for (int j = 0; j < N_COORDS; j++) {
            dist = dists[i * N_COORDS + j];
            r_min_over_dist = r_mins[j]/dist;
            // beyond cutoff
            if (dist > rc_vdw[j]) {
                cur_grid_val += eps_sqrt[j] * pow(r_min_over_dist, 12.0) -
                2.0 * pow(r_min_over_dist, 6.0);
            // within cutoff
            } else {
                cur_grid_val += vwd_softcore_max * 
                (1.0 - 0.5 * pow((dist / rc_vdw[j]), beta[j]));
            }
        }
        vdw_grid[i] = cur_grid_val;
    }
    free(r_mins);
    free(eps_sqrt);
    free(rc_vdw);
    free(beta);
}