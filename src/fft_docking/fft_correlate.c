#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include <string.h>

// Function to fill the padded 3D array for ligand grid for FFT
void fill_padded_array(
  int x, int y, int z, 
  int pad_x, int pad_y, int pad_z,
  float *arr_l, float *padded_arr_l
) {
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      float *padded_arr_l_ij = padded_arr_l + (pad_y * pad_z * i) + (pad_z * j);
      float *arr_l_ij = arr_l + (y * z * i) + z * j;
      memcpy(padded_arr_l_ij, arr_l_ij, z * sizeof(float));
    }
  }
}

// Function to roll and sum the array
void roll_and_sum_array(
  float *grids, float *result, int roll_x, int roll_y, int roll_z,
  int nx, int ny, int nz
){
  // Get array dimensions
  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        int updated_idx = (
          (x + roll_x) % nx * ny * nz + 
          (y + roll_y) % ny * nz + 
          (z + roll_z) % nz
        );
        result[updated_idx] += grids[
          x * ny * nz + y * nz + z
        ];
      }
    }
  }
}

void sum_grids(
  float *grids, float *result, size_t N_orietations, int N_grids, size_t N_scores
) {
  #pragma omp parallel for 
  for (size_t i = 0; i < N_orietations; i++) {
    for (int j = 0; j < N_grids; j++) {
      for (size_t k = 0; k < N_scores; k++) {
        result[N_scores * i + k] += grids[N_scores * (i * N_grids + j) + k];
      }
    }
  }
}

// Function to perform 3D FFT correlation
void fft_correlate(
  float *recep_arr, float *lig_arr, int N_grids, 
  // Protein receptor grid dimensions
  int nx, int ny, int nz,
  // Ligand grid dimensions
  int nx_lig, int ny_lig, int nz_lig,
  int N_orientations, int N_threads, 
  float *result_arr
) {
  // Number of grid points
  size_t N_grid_points = nx * ny * nz;
  size_t N_lig_grid_points = nx_lig * ny_lig * nz_lig;

  // Number FFT coefficients (only half of the array is needed due to symmetry)
  size_t N_fft_points = nx * ny * (nz / 2 + 1); 
  // Allocate memory for FFTW plans and data
  fftwf_plan plan_fwd, plan_inv;
  fftwf_complex *fft_r, *fft_l;
  fft_r = (fftwf_complex *)fftwf_malloc(
    sizeof(fftwf_complex) * N_fft_points * N_grids
  );
  fft_l = (fftwf_complex *)fftwf_malloc(
    sizeof(fftwf_complex) * N_fft_points * N_orientations
  );

  // Create forward and inverse FFT plans for both arrays
  plan_fwd = fftwf_plan_dft_r2c_3d(nx, ny, nz, recep_arr, fft_r, FFTW_MEASURE);
  plan_inv = fftwf_plan_dft_c2r_3d(nx, ny, nz, fft_r, recep_arr, FFTW_MEASURE);

  float scale = 1.0 / N_grid_points;
  int N_threads_recep = N_grids > N_threads ? N_threads : N_grids;
  // Execute forward FFTs on receptor arrays
  #pragma omp parallel for num_threads(N_threads_recep)
  for (int i = 0; i < N_grids; i++) {
    float *cur_recep = recep_arr + (N_grid_points * i);
    fftwf_complex *cur_fft_r = fft_r + (N_fft_points * i);
    fftwf_execute_dft_r2c(plan_fwd, cur_recep, cur_fft_r);
  }
  // Execute forward FFTs on ligand arrays and correlate with receptor arrays
  #pragma omp parallel for num_threads(N_threads)
  for (int j = 0; j < N_orientations; j++) {
    for (int i = 0; i < N_grids; i++) {
      fftwf_complex *cur_fft_r = fft_r + (N_fft_points * i);
      float *cur_result_arr = result_arr + (N_grid_points * (N_grids * j + i));
      float *padded_lig = (float *)fftwf_alloc_real(N_grid_points);
      memset(padded_lig, 0.0, N_grid_points * sizeof(float));
      float *cur_lig = lig_arr + (N_lig_grid_points * (N_grids * j + i));
      fill_padded_array(nx_lig, ny_lig, nz_lig, nx, ny, nz, cur_lig, padded_lig);
      fftwf_complex *cur_fft_l = fft_l + (N_fft_points * j);
      // The pointer to the padded ligand array is the same as the pointer 
      // to the correlation array
      fftwf_execute_dft_r2c(plan_fwd, padded_lig, cur_fft_l);
      for (size_t k = 0; k < N_fft_points; k++) {
        cur_fft_l[k] = cur_fft_r[k] * conjf(cur_fft_l[k]) * scale;
      }
      // Execute inverse FFT on the product
      fftwf_execute_dft_c2r(plan_inv, cur_fft_l, padded_lig);
      memcpy(cur_result_arr, padded_lig, N_grid_points * sizeof(float));
      fftwf_free(padded_lig);
    }
  }

  // Clean up memory and plans
  fftwf_destroy_plan(plan_fwd);
  fftwf_destroy_plan(plan_inv);
  fftwf_free(fft_r);
  fftwf_free(fft_l);
}