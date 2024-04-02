#ifndef FFT_CORRELATE_H
#define FFT_CORRELATE_H

// Function to perform 3D FFT correlation
void fft_correlate(
  float *recep_arr, float *lig_arr, int N_grids, 
  // Protein receptor grid dimensions
  int nx, int ny, int nz,
  // Ligand grid dimensions
  int nx_lig, int ny_lig, int nz_lig,
  int N_orientations, int N_threads, 
  float *result_arr
);

void sum_grids(
  float *grids, float *result, size_t N_orietations, int N_grids, size_t N_grid_points
);

#endif