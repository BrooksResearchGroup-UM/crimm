#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <Python.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>

// Function to pad 3D array
void pad_3d_array(
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

// Function to excute FFT correlation with created plans
void excute_fft_correlate(
  fftwf_plan plan_r_fwd, fftwf_plan plan_l_fwd, fftwf_plan plan_inv,
  fftwf_complex *fft_r, fftwf_complex *fft_l, fftwf_complex *fft_prod, 
  int N_grid_points, int N_fft_points, float *corr
) {
  // Execute forward FFTs on both arrays
  fftwf_execute(plan_r_fwd);
  fftwf_execute(plan_l_fwd);

  float scale = 1.0 / N_grid_points;
  // Perform element-wise complex conjugate multiplication
  for (int i = 0; i < N_fft_points; ++i) {
    fft_prod[i] = conjf(fft_r[i]) * fft_l[i] * scale;
  }

  // Execute inverse FFT on the product
  fftwf_execute(plan_inv);
}

// Function to perform 3D FFT correlation
void fft_correlate(
  PyArrayObject *recep_grid, PyArrayObject *lig_grid, PyArrayObject *result
) {
  
  // Check array dimensions and data type
  if (PyArray_NDIM(recep_grid) != 4 || PyArray_NDIM(lig_grid) != 4 ||
      PyArray_TYPE(recep_grid) != NPY_FLOAT32 || PyArray_TYPE(lig_grid) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_TypeError, "Expected arrays of float32 with 4 dimensions");
    return;
  }

  int n_grids_r = PyArray_DIMS(recep_grid)[0];
  int n_grids_l = PyArray_DIMS(lig_grid)[0];
  int n_grids_result = PyArray_DIMS(result)[0];
  if (n_grids_r != n_grids_l || n_grids_r != n_grids_result) {
    PyErr_SetString(
      PyExc_TypeError, 
      "Expected same number of grids for both receptor, ligand, and result arrays."
    );
    return;
  }
  // Get array dimensions
  int nx = PyArray_DIMS(recep_grid)[1];
  int ny = PyArray_DIMS(recep_grid)[2];
  int nz = PyArray_DIMS(recep_grid)[3];
  int N_grid_points = nx * ny * nz;
  // Number FFT coefficients (only half of the array is needed due to symmetry)
  int N_fft_points = nx * ny * (nz / 2 + 1); 

  // Get lig array dimensions
  int nx_l = PyArray_DIMS(lig_grid)[1];
  int ny_l = PyArray_DIMS(lig_grid)[2];
  int nz_l = PyArray_DIMS(lig_grid)[3];


  // Get NumPy arrays data pointers
  float *arr_r = (float *)PyArray_DATA(recep_grid);
  float *arr_l = (float *)PyArray_DATA(lig_grid);
  float *corr = (float *)PyArray_DATA(result);

  // Allocate memory for the padded array
  float *padded_arr_l = (float *)fftwf_malloc(sizeof(float) * N_grid_points);
  memset(padded_arr_l, 0, sizeof(float) * N_grid_points);
  // Copy the contents of arr_l into padded_arr_l
  pad_3d_array(nx_l, ny_l, nz_l, nx, ny, nz, arr_l, padded_arr_l);

  // Allocate memory for FFTW plans and data
  fftwf_plan plan_r_fwd, plan_l_fwd, plan_inv;
  fftwf_complex *fft_r, *fft_l, *fft_prod;
  fft_r = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);
  fft_l = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);
  fft_prod = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * N_fft_points);

  // Create forward and inverse FFT plans for both arrays
  plan_r_fwd = fftwf_plan_dft_r2c_3d(nx, ny, nz, arr_r, fft_r, FFTW_ESTIMATE);
  plan_l_fwd = fftwf_plan_dft_r2c_3d(nx, ny, nz, padded_arr_l, fft_l, FFTW_ESTIMATE);
  plan_inv = fftwf_plan_dft_c2r_3d(nx, ny, nz, fft_prod, corr, FFTW_ESTIMATE);

  // Perform FFT correlation for each grid
  for (int i = 0; i < n_grids_r; ++i) {
    // TODO: Add support for multiple grids, currently only the first grid is used
    // Need to seperate array data for each grid
    excute_fft_correlate(
      plan_r_fwd, plan_l_fwd, plan_inv, fft_r, fft_l, fft_prod, N_grid_points,
      N_fft_points, corr
    );
  }

  // Clean up memory and plans
  fftwf_destroy_plan(plan_r_fwd);
  fftwf_destroy_plan(plan_l_fwd);
  fftwf_destroy_plan(plan_inv);
  fftwf_free(fft_r);
  fftwf_free(fft_l);
  fftwf_free(padded_arr_l);
  fftwf_free(fft_prod);
}

// Wrapper function to be called from Python
static PyObject* py_fft_correlate(PyObject* self, PyObject* args) {
    PyArrayObject *recep_grid, *lig_grid, *result;
    if (!PyArg_ParseTuple(
      args, "O!O!O!", 
      &PyArray_Type, &recep_grid, 
      &PyArray_Type, &lig_grid,
      &PyArray_Type, &result
    )) {
        return NULL;
    }

    fft_correlate(recep_grid, lig_grid, result);

    Py_RETURN_NONE;
}

// Method table
static PyMethodDef FftCorrelateMethods[] = {
    {"fft_correlate", py_fft_correlate, METH_VARARGS, "FFT Correlation"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef fftcorrelatemodule = {
    PyModuleDef_HEAD_INIT,
    "fft_correlate",   // name of module
    NULL,              // module documentation, may be NULL
    -1,                // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    FftCorrelateMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_fft_correlate(void) {
    import_array();
    return PyModule_Create(&fftcorrelatemodule);
}