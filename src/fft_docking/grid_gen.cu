// This file contains the CUDA kernels for calculating pairwise distances,
// electrostatic grids and vdw grids. This version is for CUDA device of 
// compute capability above sm_6X (pascal), where atomicAdd supports 
// float precision.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper function for checking CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
        __FILE__, __LINE__, result, cudaGetErrorString(result), #call); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void calc_pairwise_dist_kernel(
    float* grid_pos, float* coords, int N_coords, int N_grid_points, 
    float* dists
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (i < N_grid_points && j < N_coords) {
        float dx, dy, dz;

        dx = grid_pos[i * 3] - coords[j * 3];
        dy = grid_pos[i * 3 + 1] - coords[j * 3 + 1];
        dz = grid_pos[i * 3 + 2] - coords[j * 3 + 2];

        dists[i * N_coords + j] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

__device__ float calc_point_elec_potential_radial(
    const float dist_sq, const float elec_const, const float charge, 
    const float rc_sq, const float alpha, const float elec_rep_max, 
    const float elec_attr_max
) {
    float cur_potential;
    // beyond cutoff
    if (dist_sq > rc_sq) {
        cur_potential = elec_const / dist_sq;
    // within cutoff
    } else {
        float alpha_tmp = alpha * dist_sq;
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

__device__ float calc_point_elec_potential_constant(
    const float dist_sq, const float elec_const, const float charge, 
    const float rc_sq, const float alpha, const float elec_rep_max, 
    const float elec_attr_max
) {
    float cur_potential, dist;
    dist = sqrt(dist_sq);
    // beyond cutoff
    if (dist_sq > rc_sq) {
        cur_potential = elec_const / dist;
    // within cutoff
    } else {
        float alpha_tmp = alpha * dist;
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

__global__ void gen_all_grid_kernel(
    const float* grid_pos, const float* coords, const float *charges, 
    const float* epsilons, const float* vdw_rs, const float cc_elec, 
    const float rad_dielec_const, const float elec_rep_max, 
    const float elec_attr_max, const float vwd_rep_max,
    const float vwd_attr_max, const int N_coords, const int N_grid_points,
    const int use_constant_dielectric,
    float* electrostat_grid, float* vdw_grid_attr, float* vdw_grid_rep
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float (*calc_point_elec_potential)(
        float, float, float, float, float, float, float
    );

    // set the electrostatic potential function
    if (use_constant_dielectric) {
        calc_point_elec_potential = &calc_point_elec_potential_constant;
    } else {
        calc_point_elec_potential = &calc_point_elec_potential_radial;
    }

    if (i < N_grid_points && j < N_coords) {
        float dx, dy, dz, dist_sq;
        float r_min, eps_sqrt, vdwconst_attr, vdwconst_rep, r6;
        float beta_attr, beta_rep;
        float rc_sq_vdw_attr, rc_sq_vdw_rep, cur_vdw_potential_attr, cur_vdw_potential_rep;
        float charge, elec_const, emax_tmp, alpha, rc_sq_elec, cur_elec_potential;
        // calculate vdw constants
        r_min = vdw_rs[j];
        eps_sqrt = sqrtf(fabs(epsilons[j]));
        vdwconst_attr = 0.5 * fabsf(vwd_attr_max) / eps_sqrt;
        vdwconst_rep = 0.25 * fabsf(vwd_rep_max) / eps_sqrt;
        // vdw cutoff distance squared
        rc_sq_vdw_attr = r_min * r_min * powf(vdwconst_attr, -1.0 / 3.0);
        rc_sq_vdw_rep = r_min * r_min * powf(vdwconst_rep, -1.0 / 6.0);
        beta_attr = -12.0 * eps_sqrt / vwd_attr_max * vdwconst_attr;
        beta_rep = 12.0 * eps_sqrt / vwd_rep_max * vdwconst_rep;
        // calculate electrostatic constants
        charge = charges[j];
        elec_const = cc_elec * charge / rad_dielec_const;
        if (charge > 0) {
            // repulsive
            emax_tmp = elec_rep_max;
        } else {
            // attractive
            emax_tmp = elec_attr_max;
        }
        // elec cutoff distance squared
        rc_sq_elec = 2.0 * fabsf(elec_const / emax_tmp);
        alpha = fabsf(emax_tmp / (2.0 * rc_sq_elec));

        // pairwise distances squared between the receptor atom coords 
        // and the grid points
        dx = grid_pos[i * 3] - coords[j * 3];
        dy = grid_pos[i * 3 + 1] - coords[j * 3 + 1];
        dz = grid_pos[i * 3 + 2] - coords[j * 3 + 2];
        dist_sq = dx * dx + dy * dy + dz * dz;
        
        r6 = powf(r_min / dist_sq, 3.0);
        cur_vdw_potential_attr = -2 * eps_sqrt * r6;
        // attractive vdw grid
        if (dist_sq > rc_sq_vdw_attr) {
            // beyond cutoff
            cur_vdw_potential_attr = eps_sqrt * r6 * r6;
        } else {
            // within cutoff
            cur_vdw_potential_attr = vwd_attr_max * 
            (1.0 - 0.5 * powf(dist_sq / rc_sq_vdw_attr, beta_rep));
        }
        atomicAdd(&vdw_grid_attr[i], cur_vdw_potential_attr); 
        // repulsive vdw grid
        if (dist_sq > rc_sq_vdw_rep) {
            // beyond cutoff
            cur_vdw_potential_rep = eps_sqrt * r6 * r6;
        } else {
            // within cutoff
            cur_vdw_potential_rep = vwd_rep_max * 
            (1.0 - 0.5 * powf(dist_sq / rc_sq_vdw_rep, beta_attr));
        }
        atomicAdd(&vdw_grid_rep[i], cur_vdw_potential_rep); 
        cur_elec_potential = (*calc_point_elec_potential)(
                dist_sq, elec_const, charge, rc_sq_elec, alpha,
                elec_rep_max, elec_attr_max
        );
        atomicAdd(&electrostat_grid[i], cur_elec_potential); 
    }
}

void calc_chunk_pairwise_dist(
    float* host_grid_pos, float* host_coords, 
    const int N_coords, const int N_grid_points, size_t chunk_size,
    size_t num_chunks, float* host_dists
) {
    float* device_grid_pos;
    float* device_coords;
    float* device_dists;

    cudaMalloc((void**)&device_coords, N_coords * 3 * sizeof(float));
    cudaMemcpy(
        device_coords, host_coords,
        N_coords * 3 * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMalloc((void**)&device_grid_pos, chunk_size * 3 * sizeof(float));
    CUDA_CHECK(cudaPeekAtLastError());
    cudaMalloc((void**)&device_dists, chunk_size * N_coords * sizeof(float));
    CUDA_CHECK(cudaPeekAtLastError());

    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min<size_t>((i + 1) * chunk_size, N_grid_points);
        size_t cur_chunk_size = end - start;
        printf("[Chunk %zu] Num of grid points: %zu\n", i, cur_chunk_size);
        
        // Copy chunk to device
        cudaMemcpy(
            device_grid_pos, &host_grid_pos[start * 3], 
            cur_chunk_size * 3 * sizeof(float), 
            cudaMemcpyHostToDevice
        );
        CUDA_CHECK(cudaPeekAtLastError());
        // cudaMemset(device_dists, 0, chunk_size * N_coords * sizeof(float));
        // Run the kernel
        dim3 dimBlock(32, 32);
        dim3 dimGrid(
            (cur_chunk_size + dimBlock.x - 1) / dimBlock.x,
            (N_coords + dimBlock.y - 1) / dimBlock.y
        );
        calc_pairwise_dist_kernel<<<dimGrid, dimBlock>>>(
            device_grid_pos, device_coords,
            N_coords, cur_chunk_size, device_dists
        );
        CUDA_CHECK(cudaPeekAtLastError());
        cudaMemcpy(
            &host_dists[start * N_coords], device_dists,
            cur_chunk_size * N_coords * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        CUDA_CHECK(cudaPeekAtLastError());
    }
    cudaFree(device_dists);
    cudaFree(device_grid_pos);
    cudaFree(device_coords);
}

extern "C"
void cuda_calc_all_pairwise_dist(
    float* host_grid_pos, float* host_coords, 
    const int N_coords, const int N_grid_points, float* host_dists
) {
    // Allocate memory on the device
    float* device_grid_pos;
    float* device_coords;
    float* device_dists;

    cudaMalloc((void**)&device_grid_pos, N_grid_points * 3 * sizeof(float));
    cudaMalloc((void**)&device_coords, N_coords * 3 * sizeof(float));
    cudaMalloc((void**)&device_dists, N_grid_points * N_coords * sizeof(float));

    // Copy data to the device
    cudaMemcpy(
        device_grid_pos, host_grid_pos, 
        N_grid_points * 3 * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_coords, host_coords, 
        N_coords * 3 * sizeof(float), 
        cudaMemcpyHostToDevice
    );

    // Run the kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (N_grid_points + dimBlock.x - 1) / dimBlock.x,
        (N_coords + dimBlock.y - 1) / dimBlock.y
    );
    calc_pairwise_dist_kernel<<<dimGrid, dimBlock>>>(
        device_grid_pos, device_coords, N_coords, N_grid_points, device_dists
    );

    // Copy data back to the host
    cudaMemcpy(
        host_dists, device_dists, 
        N_grid_points * N_coords * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    // Free memory on the device
    cudaFree(device_grid_pos);
    cudaFree(device_coords);
    cudaFree(device_dists);
}

extern "C"
void cuda_calc_pairwise_dist(
    float* host_grid_pos, float* host_coords, 
    const int N_coords, const int N_grid_points, float* host_dists
){
    size_t dists_size = N_grid_points * N_coords * sizeof(float);
    size_t coords_size = N_coords * 3 * sizeof(float);
    size_t grid_pos_size = N_grid_points * 3 * sizeof(float);

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    size_t device_memory = prop.totalGlobalMem;
    size_t leftover_memory = device_memory - coords_size - grid_pos_size;

    if (dists_size > leftover_memory){
        // 1.1 is a safety factor for 10% headroom
        size_t chunk_size = leftover_memory / (N_coords * 1.1 * sizeof(float));
        size_t num_chunks = (N_grid_points + chunk_size - 1) / chunk_size;
        printf(
            "Array size is too large for the device memory (%zuMB). Split into %zu chunks\n", 
            leftover_memory/1024/1024, num_chunks
        );
        calc_chunk_pairwise_dist(
            host_grid_pos, host_coords, 
            N_coords, N_grid_points, chunk_size, num_chunks, host_dists
        );
    } else {
        calc_all_pairwise_dist(
            host_grid_pos, host_coords, 
            N_coords, N_grid_points, host_dists);
    }
}


extern "C"
void cuda_gen_all_grids(
    const float* host_grid_pos, const float* host_coords, const float* host_charges, 
    const float* host_epsilons, const float* host_vdw_rs, const float cc_elec, 
    const float rad_dielec_const, const float elec_rep_max, 
    const float elec_attr_max,
    const float vwd_softcore_max, const int N_coords, const int N_grid_points,
    const int use_constant_dielectric,
    float* host_electrostat_grid, float* host_vdw_grid_attr, float* host_vdw_grid_rep
){
    // Allocate memory on the device
    // input
    float* device_grid_pos;
    float* device_coords;
    float* device_charges;
    float* device_epsilons;
    float* device_vdw_rs;
    // output
    float* device_electrostat_grid;
    float* device_vdw_grid_attr;
    float* device_vdw_grid_rep;
    // input
    cudaMalloc((void**)&device_grid_pos, N_grid_points * 3 * sizeof(float));
    cudaMalloc((void**)&device_coords, N_coords * 3 * sizeof(float));
    cudaMalloc((void**)&device_charges, N_coords * sizeof(float));
    cudaMalloc((void**)&device_epsilons, N_coords * sizeof(float));
    cudaMalloc((void**)&device_vdw_rs, N_coords * sizeof(float));
    // output
    cudaMalloc((void**)&device_electrostat_grid, N_grid_points * sizeof(float));
    cudaMalloc((void**)&device_vdw_grid_attr, N_grid_points * sizeof(float));
    cudaMalloc((void**)&device_vdw_grid_rep, N_grid_points * sizeof(float));


    // Copy data to the device
    cudaMemcpy(
        device_grid_pos, host_grid_pos, 
        N_grid_points * 3 * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_coords, host_coords, 
        N_coords * 3 * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_charges, host_charges, 
        N_coords * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_epsilons, host_epsilons, 
        N_coords * sizeof(float), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_vdw_rs, host_vdw_rs, 
        N_coords * sizeof(float), 
        cudaMemcpyHostToDevice
    );

    // Run the kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        (N_grid_points + dimBlock.x - 1) / dimBlock.x,
        (N_coords + dimBlock.y - 1) / dimBlock.y
    );
    gen_all_grid_kernel<<<dimGrid, dimBlock>>>(
        device_grid_pos, device_coords, device_charges, device_epsilons, 
        device_vdw_rs, cc_elec, rad_dielec_const, elec_rep_max, elec_attr_max,
        vwd_softcore_max, N_coords, N_grid_points, use_constant_dielectric,
        device_electrostat_grid, device_vdw_grid_attr, device_vdw_grid_rep
    ); 

    // Copy data back to the host
    cudaMemcpy(
        host_electrostat_grid, device_electrostat_grid, 
        N_grid_points * sizeof(float), 
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        host_vdw_grid_rep, device_vdw_grid_rep, 
        N_grid_points * sizeof(float), 
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        host_vdw_grid_attr, device_vdw_grid_attr, 
        N_grid_points * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    // Free memory on the device
    cudaFree(device_grid_pos);
    cudaFree(device_coords);
    cudaFree(device_charges);
    cudaFree(device_epsilons);
    cudaFree(device_vdw_rs);
    cudaFree(device_electrostat_grid);
    cudaFree(device_vdw_grid_attr);
    cudaFree(device_vdw_grid_rep);
}